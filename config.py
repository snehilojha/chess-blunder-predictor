from pathlib import Path

ROOT = Path(__file__).resolve().parent

# paths
DATA_DIR = ROOT / "data"
RAW_PGN_PATH = DATA_DIR / "raw" / "lichess_db_standard_rated_2017-06.pgn"
RAW_GAMES_PATH = DATA_DIR / "processed" / "games.parquet"         # output of data_loader
CLEANED_PATH = DATA_DIR / "processed" / "cleaned_games.parquet"   # output of preprocessor
MODELS_DIR = ROOT / "models"
METADATA_PATH = MODELS_DIR / "preprocessing_metadata.json"
FEATURES_PATH = DATA_DIR / "processed" / "features.parquet"

# Constants
SNAPSHOT_MOVE = 15
BLUNDER_THRESHOLD = 200
BLUNDER_WINDOW = 10

# Data filtering
MIN_ELO = 1000
MAX_ELO = 2500
TIME_CONTROLS = [600, 900, 1800] # 10, 15, 30 minutes
CAP = 1000

FEATURE_COLS = [
    "player_elo", "elo_diff", "time_control",
    "eval_at_snapshot", "eval_volatility", "eval_trend",
    "material_balance", "material_imbalance", "king_attackers", "has_castled"
]

# XGBoost parameters
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 5.82, # recalculated after per-side label split (~14% blunder rate)
    "eval_metric": "auc",
    "random_state": 42,
}

N_CV_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# MLflow
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlruns.db'}"
MLFLOW_ARTIFACT_URI = (ROOT / "models" / "mlflow_artifacts").as_uri()
MLFLOW_EXPERIMENT = 'chess_blunder_prediction'
MLFLOW_MODEL_NAME = 'blunder_predictor'
MODEL_VERSION = 1
