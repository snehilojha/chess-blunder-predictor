import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import chess.pgn
import pandas as pd
from config import (
    RAW_PGN_PATH, RAW_GAMES_PATH,
    MIN_ELO, MAX_ELO, TIME_CONTROLS
)

MAX_GAMES = 15000

def has_evals(game: chess.pgn.Game) -> bool:
    """Check if a game has Stockfish eval annotations on its moves."""
    node = game
    while node.variations:
        node = node.variation(0)
        if node.eval() is not None:
            return True
    return False

def passes_filters(game: chess.pgn.Game) -> bool:
    """Return True if game meets Elo range and time control requirement."""
    headers = game.headers
    try:
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
    except ValueError:
        return False
    
    if not (MIN_ELO <= white_elo <= MAX_ELO and MIN_ELO <= black_elo <= MAX_ELO):
        return False
    
    time_control = headers.get("TimeControl", "")
    base_time = time_control.split("+")[0]
    if base_time not in TIME_CONTROLS:
        return False
    
    return True

def extract_game_record(game: chess.pgn.Game) -> dict:
    """
    Extract flat metadata from a game object.
    Returns one record per game - per-side split happens in feature engineering.
    """
    headers = game.headers
    moves = []
    evals = []

    node = game
    while node.variations:
        node = node.variation(0)
        moves.append(node.move.uci())
        ev = node.eval()
        if ev is not None:
            if ev.is_mate():
                cp = 10000 if ev.relative.mate() > 0 else -10000
            else:
                cp = ev.relative.score()
        
        else:
            cp = None
        evals.append(cp)

    return {
        "game_id": headers.get("Site", ""),
        "white_elo": int(headers.get("WhiteElo", 0)),
        "black_elo": int(headers.get("BlackElo", 0)),
        "time_control": headers.get("TimeControl", ""),
        "opening": headers.get("Opening", ""),
        "result": headers.get("Result", ""),
        "moves": moves,
        "evals": evals,
    }


def load_games(pgn_path: Path = RAW_PGN_PATH, max_games: int = MAX_GAMES) -> pd.DataFrame:
    """
    Stream PGN file, apply filters, collect up to max_games valid games.
    Returns a DataFrame with one row per game containing raw moves and evals.
    """
    records = []
    games_seen = 0
    games_collected = 0
 
    print(f"Streaming {pgn_path.name} — target: {max_games} valid games")
 
    with open(pgn_path, encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
 
            games_seen += 1
 
            if games_seen % 10_000 == 0:
                print(f"  Scanned {games_seen:,} games — collected {games_collected:,}")
 
            if not passes_filters(game):
                continue
            if not has_evals(game):
                continue
 
            records.append(extract_game_record(game))
            games_collected += 1
 
            if games_collected >= max_games:
                break
 
    print(f"\nDone. Scanned {games_seen:,} games, collected {games_collected:,}.")
    return pd.DataFrame(records)
 
 
def save_processed(df: pd.DataFrame, path: Path = RAW_GAMES_PATH) -> None:
    """Save DataFrame to parquet. Creates directory if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} games to {path}")
 
 
if __name__ == "__main__":
    df = load_games()
    print(df.head())
    print(f"\nShape: {df.shape}")
    save_processed(df)
