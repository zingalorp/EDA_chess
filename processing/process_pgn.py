import chess
import chess.pgn
import re
import time
import gzip
import bz2
import zstandard as zstd
import io
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, List, Iterator

# --- Configuration ---
INPUT_PGN_FILES = [
    Path(r"E:\EDA_chess_data\lichess_db_standard_rated_2018-01.pgn.zst"),
    Path(r"E:\EDA_chess_data\lichess_db_standard_rated_2019-01.pgn.zst"),
    Path(r"E:\EDA_chess_data\lichess_db_standard_rated_2020-01.pgn.zst"),
    Path(r"E:\EDA_chess_data\lichess_db_standard_rated_2021-01.pgn.zst"),
    Path(r"E:\EDA_chess_data\lichess_db_standard_rated_2022-01.pgn.zst"),
    Path(r"E:\EDA_chess_data\lichess_db_standard_rated_2023-01.pgn.zst"),
    Path(r"E:\EDA_chess_data\lichess_db_standard_rated_2024-01.pgn.zst"),
    Path(r"E:\EDA_chess_data\lichess_db_standard_rated_2025-01.pgn.zst"),
]
OUTPUT_PARQUET_DIR = Path(r"E:\EDA_chess_data\processed")
GAMES_PER_FILE_LIMIT = 20000
WRITE_BATCH_SIZE = 50000  # Number of moves to accumulate before writing

# --- Helper Functions ---

def open_decompressed_stream(filepath: Path):
    """Opens a potentially compressed file based on extension."""
    filepath = Path(filepath)  # Ensure it's a Path object
    mode = 'rt'  # Read text
    encoding = 'utf-8'  # Standard for PGN

    if filepath.suffix == '.gz':
        return gzip.open(filepath, mode, encoding=encoding, errors='replace')
    elif filepath.suffix == '.bz2':
        return bz2.open(filepath, mode, encoding=encoding, errors='replace')
    elif filepath.suffix == '.zst':
        # zstandard needs bytes stream first, then decode
        fh = open(filepath, 'rb')
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream_reader, encoding=encoding, errors='replace')
    else:  # Assume uncompressed
        return open(filepath, mode, encoding=encoding, errors='replace')

def calculate_material(board: chess.Board) -> int:
    """Calculates material balance (White perspective: White_Material - Black_Material)."""
    white_material = 0
    black_material = 0
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    for square, piece in board.piece_map().items():
        value = piece_values.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            white_material += value
        else:
            black_material += value
    return white_material - black_material

def determine_game_phase(board: chess.Board, ply: int) -> str:
    """Determines game phase based on piece counts and board state."""
    major_minor_pieces: set[chess.PieceType] = {
        chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN
    }
    major_minor_piece_count = 0
    white_pieces_on_opponent_half = 0
    black_pieces_on_opponent_half = 0
    for square, piece in board.piece_map().items():
        if piece.piece_type in major_minor_pieces:
            major_minor_piece_count += 1
        if piece.piece_type not in (chess.PAWN, chess.KING):
            rank = chess.square_rank(square)
            if piece.color == chess.WHITE and rank >= 4:
                white_pieces_on_opponent_half += 1
            elif piece.color == chess.BLACK and rank <= 3:
                black_pieces_on_opponent_half += 1

    if major_minor_piece_count <= 6:
        return 'Endgame'
    if major_minor_piece_count <= 10:
        return 'Middlegame'

    if not board.has_kingside_castling_rights(chess.WHITE) and \
       not board.has_queenside_castling_rights(chess.WHITE):
        return 'Middlegame'
    if not board.has_kingside_castling_rights(chess.BLACK) and \
       not board.has_queenside_castling_rights(chess.BLACK):
        return 'Middlegame'

    if white_pieces_on_opponent_half >= 2 or black_pieces_on_opponent_half >= 2:
         return 'Middlegame'

    return 'Opening'

def find_pgn_comment(comment: str, tag: str) -> Optional[str]:
    """Extracts value from PGN comments like [%tag value]."""
    pattern = rf'\[%{tag}\s+([^\]]+)\]'
    match = re.search(pattern, comment)
    if match:
        return match.group(1).strip()
    return None

def parse_clock_to_seconds(clk_str: Optional[str], default: Optional[float] = None) -> Optional[float]:
    """Converts H:MM:SS.ms or MM:SS.ms clock string to seconds."""
    if clk_str is None:
        return default
    parts = clk_str.split(':')
    try:
        if len(parts) == 3:
            return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0])*60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
        else:
            return default
    except ValueError:
        return default

def safe_float_convert(value_str: Optional[str], default: Optional[float] = None) -> Optional[float]:
    """Safely converts a string to float, handling None and errors."""
    if value_str is None:
        return default
    try:
        if value_str.startswith('#'):
             mate_score = 10000.0
             try:
                 mate_in_plies = int(value_str[1:])
                 return mate_score if mate_in_plies > 0 else -mate_score
             except ValueError:
                 return default  # Malformed mate
        return float(value_str)
    except ValueError:
        return default

def get_player_elo(headers: chess.pgn.Headers, color: chess.Color) -> Optional[int]:
    """Safely gets ELO from headers."""
    elo_tag = 'WhiteElo' if color == chess.WHITE else 'BlackElo'
    elo_str = headers.get(elo_tag, None)
    if elo_str is None or elo_str == '?':
        return None
    try:
        return int(elo_str)
    except ValueError:
        return None

def parse_time_control(tc_string: str) -> Tuple[Optional[int], Optional[int]]:
    """Parses 'Initial+Increment' time control string."""
    if '+' in tc_string:
        try:
            initial_str, increment_str = tc_string.split('+', 1)
            return int(initial_str), int(increment_str)
        except ValueError:
            return None, None
    elif tc_string.isdigit():
         try:
             return int(tc_string), 0
         except ValueError:
             return None, None
    else:
         return None, None

# --- Main Processing Function for a Single Game ---

def process_game(game: chess.pgn.Game, file_year: int) -> List[Dict[str, Any]]:
    """Processes a single game and returns a list of dictionaries, one per move.
       The year is taken from the PGN file name (file_year) rather than the PGN headers."""
    game_move_data = []
    try:
        board = game.board()
        headers = game.headers

        # --- Use file_year extracted from the filename ---
        game_year = file_year
        # We still try to parse month from the header if needed
        utc_date_str = headers.get('UTCDate', '????.??.??')
        try:
            game_month = int(utc_date_str.split('.')[1]) if utc_date_str != '????.??.??' else None
        except (ValueError, IndexError):
            game_month = None

        white_elo = get_player_elo(headers, chess.WHITE)
        black_elo = get_player_elo(headers, chess.BLACK)
        result = headers.get('Result', '*')
        termination = headers.get('Termination', 'Unknown')
        time_control_str = headers.get('TimeControl', '-')
        initial_time, increment = parse_time_control(time_control_str)
        eco_opening = headers.get('ECO', '?')

        # --- Initialize Move-Level State ---
        prev_white_clk = float(initial_time) if initial_time is not None else None
        prev_black_clk = float(initial_time) if initial_time is not None else None
        prev_eval = None

        # --- Iterate Through Moves ---
        for node in game.mainline():
            move = node.move
            if move is None:
                continue

            # --- Capture State BEFORE Move ---
            turn_color = board.turn
            fen_before = board.fen()
            ply = board.ply()
            player_elo = white_elo if turn_color == chess.WHITE else black_elo
            opponent_elo = black_elo if turn_color == chess.WHITE else white_elo
            is_check_before = board.is_check()
            try:
                num_legal_moves = board.legal_moves.count()
            except Exception:
                num_legal_moves = None
            material_balance_abs = calculate_material(board)
            player_material_balance = material_balance_abs if turn_color == chess.WHITE else -material_balance_abs
            game_phase = determine_game_phase(board, ply)
            clock_before_move = prev_white_clk if turn_color == chess.WHITE else prev_black_clk

            # --- Extract Annotations ---
            comment = node.comment
            current_clk_str = find_pgn_comment(comment, 'clk')
            current_eval_str = find_pgn_comment(comment, 'eval')
            current_clk_seconds = parse_clock_to_seconds(current_clk_str, default=None)
            current_eval = safe_float_convert(current_eval_str, default=None)

            # --- Calculate Derived Features ---
            time_spent = None
            if clock_before_move is not None and current_clk_seconds is not None:
                 time_spent = clock_before_move - current_clk_seconds
                 if time_spent < 0:
                     time_spent = 0.0

            player_relative_eval_change = None
            if current_eval is not None and prev_eval is not None:
                 eval_delta = current_eval - prev_eval
                 player_relative_eval_change = eval_delta if turn_color == chess.WHITE else -eval_delta

            # --- Get Move Info ---
            try:
                move_san = board.san(move)
            except ValueError:
                move_san = "Invalid?"
            is_capture = board.is_capture(move)
            is_castling = board.is_castling(move)

            # --- Assemble Record ---
            record = {
                'GameID': headers.get('Site', f"UnknownSite_{headers.get('Date', '')}_{headers.get('White', '')}"),
                'Year': game_year,
                'Month': game_month,
                'WhiteElo': white_elo, 
                'BlackElo': black_elo,
                'Result': result,
                'Termination': termination, 
                'TimeControl': time_control_str,
                'InitialTimeSec': initial_time, 
                'IncrementSec': increment,
                'ECO': eco_opening,
                'Ply': ply, 
                'FENBeforeMove': fen_before,
                'PlayerColor': 'White' if turn_color == chess.WHITE else 'Black',
                'PlayerELO': player_elo, 
                'OpponentELO': opponent_elo,
                'ClockTimeBeforeMoveSec': clock_before_move,
                'EvaluationBeforeMove': prev_eval,
                'MaterialBalanceAbs': material_balance_abs,
                'PlayerMaterialBalance': player_material_balance,
                'NumLegalMoves': num_legal_moves,
                'IsCheck_BeforeMove': is_check_before,
                'GamePhase': game_phase,
                'MoveSAN': move_san,
                'IsCapture': is_capture,
                'IsCastling': is_castling,
                'ClockTimeAfterMoveSec': current_clk_seconds,
                'EvaluationAfterMove': current_eval,
                'TimeSpentSec': time_spent,
                'PlayerRelativeEvalChange': player_relative_eval_change,
                'IsBlunder_100cp': player_relative_eval_change < -1.0 if player_relative_eval_change is not None else None,
            }
            game_move_data.append(record)

            # --- Push Move & Update State ---
            board.push(move)
            if current_clk_seconds is not None:
                if turn_color == chess.WHITE:
                    prev_white_clk = current_clk_seconds
                else:
                    prev_black_clk = current_clk_seconds
            if current_eval is not None:
                prev_eval = current_eval

    except Exception as e:
        error_game_id = headers.get('Site', 'Unknown') if 'headers' in locals() else 'Unknown Game'
        print(f"\nERROR processing game {error_game_id}: {type(e).__name__} - {e}. Skipping game.")
        return []  # Return empty list on error

    return game_move_data

# --- Main Processing Loop ---

total_moves_processed = 0
is_first_write = True
parquet_schema = None  # Store schema after first write

start_time_total = time.time()

for pgn_filepath in INPUT_PGN_FILES:
    # Extract the year from the file name using a regex pattern (expecting a pattern like "2018-")
    match = re.search(r'(\d{4})-', pgn_filepath.name)
    if match:
        file_year = int(match.group(1))
    else:
        print(f"Could not extract year from file name: {pgn_filepath.name}. Skipping file.")
        continue

    print(f"\n--- Processing file: {pgn_filepath.name} (Year: {file_year}) ---")
    start_time_file = time.time()
    games_processed_in_file = 0
    moves_in_file_batch = []

    try:
        with open_decompressed_stream(pgn_filepath) as pgn_stream:
            # Use tqdm for progress bar over games in this file
            game_iterator = (chess.pgn.read_game(pgn_stream) for _ in range(GAMES_PER_FILE_LIMIT))
            progress_bar = tqdm(game_iterator,
                                total=GAMES_PER_FILE_LIMIT,
                                desc=f"Games in {pgn_filepath.name}",
                                unit="game")

            for game in progress_bar:
                if game is None:
                    print(f"\nEnd of file reached early in {pgn_filepath.name}")
                    break  # End of file

                game_moves = process_game(game, file_year)
                if game_moves:  # Only add if processing didn't fail
                    moves_in_file_batch.extend(game_moves)
                    games_processed_in_file += 1
                    total_moves_processed += len(game_moves)

                    # Check if batch needs writing
                    if len(moves_in_file_batch) >= WRITE_BATCH_SIZE:
                        # print(f"\nWriting batch of {len(moves_in_file_batch)} moves...")
                        batch_df = pd.DataFrame(moves_in_file_batch)
                        table = pa.Table.from_pandas(batch_df, schema=parquet_schema, preserve_index=False)
                        if is_first_write:
                            parquet_schema = table.schema  # Capture schema from first batch
                            is_first_write = False

                        pq.write_to_dataset(
                            table,
                            root_path=OUTPUT_PARQUET_DIR,
                            partition_cols=['Year'],  # Partition only by Year
                            schema=parquet_schema,
                            basename_template=f"part-{time.time_ns()}-{{i}}.parquet",
                            existing_data_behavior='overwrite_or_ignore'
                        )
                        # print(f"Batch written. Total moves processed so far: {total_moves_processed}")
                        moves_in_file_batch = []  # Clear batch

            progress_bar.close()

    except Exception as e:
        print(f"\nFATAL ERROR reading or processing file {pgn_filepath.name}: {type(e).__name__} - {e}")

    # Write any remaining moves from the last batch of this file
    if moves_in_file_batch:
        print(f"\nWriting final batch of {len(moves_in_file_batch)} moves for {pgn_filepath.name}...")
        batch_df = pd.DataFrame(moves_in_file_batch)
        if is_first_write and not batch_df.empty:
             temp_table = pa.Table.from_pandas(batch_df, preserve_index=False)
             parquet_schema = temp_table.schema
             is_first_write = False

        if not batch_df.empty and parquet_schema:
            table = pa.Table.from_pandas(batch_df, schema=parquet_schema, preserve_index=False)
            pq.write_to_dataset(
                table,
                root_path=OUTPUT_PARQUET_DIR,
                partition_cols=['Year'],  # Partition only by Year
                schema=parquet_schema,
                basename_template=f"part-{time.time_ns()}-{{i}}.parquet"
            )
            print(f"Final batch for file written. Total moves processed so far: {total_moves_processed}")
        elif not batch_df.empty:
             print("Warning: Could not write final batch as schema was not determined (maybe no data processed yet?).")

    file_duration = time.time() - start_time_file
    print(f"--- Finished processing {pgn_filepath.name} ---")
    print(f"Games processed in this file: {games_processed_in_file}")
    print(f"Time taken for this file: {file_duration:.2f} seconds")

total_duration = time.time() - start_time_total
print("\n=== Preprocessing Complete ===")
print(f"Total files processed: {len(INPUT_PGN_FILES)}")
print(f"Total moves extracted: {total_moves_processed}")
print(f"Dataset saved to: {OUTPUT_PARQUET_DIR}")
print(f"Total time taken: {total_duration:.2f} seconds")
