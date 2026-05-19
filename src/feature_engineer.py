import sys
import numpy as np
import pandas as pd
import chess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SNAPSHOT_MOVE, BLUNDER_THRESHOLD, BLUNDER_WINDOW, CAP

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
}


def side_has_blunder(evals, offset, start=SNAPSHOT_MOVE, end=SNAPSHOT_MOVE + BLUNDER_WINDOW, threshold=BLUNDER_THRESHOLD):
    side = [min(max(v, -CAP), CAP) for v in evals[start:end][offset::2] if v == v]
    for i in range(1, len(side)):
        if side[i-1] - side[i] > threshold:
            return 1
    return 0


def eval_features(evals, offset, color, snapshot=SNAPSHOT_MOVE):
    side = [min(max(v, -CAP), CAP) for v in evals[:snapshot][offset::2] if v == v]
    if len(side) == 0:
        return {"eval_at_snapshot": np.nan, "eval_volatility": np.nan, "eval_trend": np.nan}

    sign = 1 if color == chess.WHITE else -1
    recent = side[-5:]
    trend = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0.0

    return {
        "eval_at_snapshot": sign * side[-1],
        "eval_volatility": float(np.std(side)),
        "eval_trend": sign * trend,
    }


def board_features(moves, color, snapshot=SNAPSHOT_MOVE):
    board = chess.Board()
    castling_moves = {"e1g1", "e1c1"} if color == chess.WHITE else {"e8g8", "e8c8"}
    has_castled = 0

    for move in moves[:snapshot]:
        if move in castling_moves:
            has_castled = 1
        board.push(chess.Move.from_uci(move))

    my_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == color)
    opp_material = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color != color)
    king_sq = board.king(color)

    return {
        "material_balance": my_material - opp_material,
        "material_imbalance": abs(my_material - opp_material),
        "king_attackers": len(board.attackers(not color, king_sq)),
        "has_castled": has_castled,
    }


def build_features(df):
    df = df.copy()
    df["white_blunder"] = df["evals"].apply(lambda e: side_has_blunder(e, offset=0))
    df["black_blunder"] = df["evals"].apply(lambda e: side_has_blunder(e, offset=1))

    records = []
    for _, row in df.iterrows():
        for color, offset, elo_col, opp_col, blunder_col in [
            (chess.WHITE, 0, "white_elo", "black_elo", "white_blunder"),
            (chess.BLACK, 1, "black_elo", "white_elo", "black_blunder"),
        ]:
            ef = eval_features(row["evals"], offset, color)
            bf = board_features(row["moves"], color)
            records.append({
                "game_id": row["game_id"],
                "side": "white" if color == chess.WHITE else "black",
                "player_elo": row[elo_col],
                "elo_diff": row[elo_col] - row[opp_col],
                "time_control": row["time_control"],
                **ef,
                **bf,
                "blunder_label": row[blunder_col],
            })

    return pd.DataFrame(records)
