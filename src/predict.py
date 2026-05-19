import sys
import pandas as pd
import chess
import mlflow
import mlflow.xgboost
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME, MODEL_VERSION, FEATURE_COLS
try:
    from feature_engineer import eval_features, board_features
except ModuleNotFoundError:
    from src.feature_engineer import eval_features, board_features
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
_model = None
def get_model():
    global _model
    if _model is None:
        _model = mlflow.xgboost.load_model(f"models:/{MLFLOW_MODEL_NAME}/{MODEL_VERSION}")
    return _model
def predict(
    moves: list[str],
    evals: list[float],
    white_elo: int,
    black_elo: int,
    time_control: int,
) -> dict:
    rows = []
    for color, offset, player_elo, opp_elo in [
        (chess.WHITE, 0, white_elo, black_elo),
        (chess.BLACK, 1, black_elo, white_elo),
    ]:
        ef = eval_features(evals, offset, color)
        bf = board_features(moves, color)
        rows.append({
            "player_elo": player_elo,
            "elo_diff": player_elo - opp_elo,
            "time_control": time_control,
            **ef,
            **bf,
        })
    X = pd.DataFrame(rows, index=["white", "black"])[FEATURE_COLS]
    probas = get_model().predict_proba(X)[:, 1]
    return {
        "white_blunder_prob": round(float(probas[0]), 4),
        "black_blunder_prob": round(float(probas[1]), 4),
    }