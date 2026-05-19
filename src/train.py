import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    CLEANED_PATH, FEATURES_PATH, MODELS_DIR, MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT, MLFLOW_MODEL_NAME, XGBOOST_PARAMS,
    FEATURE_COLS, N_CV_FOLDS, TEST_SIZE, RANDOM_STATE
)
from feature_engineer import build_features


def load_features():
    if FEATURES_PATH.exists():
        return pd.read_parquet(FEATURES_PATH)
    df = pd.read_parquet(CLEANED_PATH)
    features = build_features(df)
    features.to_parquet(FEATURES_PATH, index=False)
    return features


def split(df):
    game_ids = df["game_id"].unique()
    train_ids, test_ids = train_test_split(game_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df = df[df["game_id"].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df["game_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


def run_cv(train_df):
    game_ids_train = train_df["game_id"].unique()
    game_labels = np.array([
        train_df[train_df["game_id"] == gid]["blunder_label"].max()
        for gid in game_ids_train
    ])

    kf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_aucs, cv_precs = [], []

    for fold_train_idx, fold_val_idx in kf.split(game_ids_train, game_labels):
        fold_train_ids = game_ids_train[fold_train_idx]
        fold_val_ids   = game_ids_train[fold_val_idx]

        X_fold_train = train_df[train_df["game_id"].isin(fold_train_ids)][FEATURE_COLS]
        y_fold_train = train_df[train_df["game_id"].isin(fold_train_ids)]["blunder_label"]
        X_fold_val   = train_df[train_df["game_id"].isin(fold_val_ids)][FEATURE_COLS]
        y_fold_val   = train_df[train_df["game_id"].isin(fold_val_ids)]["blunder_label"]

        m = xgb.XGBClassifier(**XGBOOST_PARAMS, verbosity=0)
        m.fit(X_fold_train, y_fold_train)
        proba = m.predict_proba(X_fold_val)[:, 1]

        cv_aucs.append(roc_auc_score(y_fold_val, proba))
        cv_precs.append(average_precision_score(y_fold_val, proba))

    return np.array(cv_aucs), np.array(cv_precs)


def train(train_df, test_df):
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["blunder_label"]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df["blunder_label"]

    cv_aucs, cv_precs = run_cv(train_df)
    print(f"CV ROC-AUC:  {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
    print(f"CV Avg Prec: {cv_precs.mean():.4f} ± {cv_precs.std():.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        model = xgb.XGBClassifier(**XGBOOST_PARAMS, verbosity=0)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc  = roc_auc_score(y_test, y_pred_proba)
        avg_prec = average_precision_score(y_test, y_pred_proba)

        mlflow.log_params(XGBOOST_PARAMS)
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("avg_precision", avg_prec)
        mlflow.log_metric("cv_roc_auc_mean", cv_aucs.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_aucs.std())
        mlflow.log_metric("cv_avg_prec_mean", cv_precs.mean())
        mlflow.log_metric("cv_avg_prec_std", cv_precs.std())
        mlflow.xgboost.log_model(model, name="model", registered_model_name=MLFLOW_MODEL_NAME)

        print(f"ROC-AUC:       {roc_auc:.4f}")
        print(f"Avg Precision: {avg_prec:.4f}")
        print(f"Run ID: {run.info.run_id}")

    return model, X_test, y_test, y_pred_proba


if __name__ == "__main__":
    df = load_features()
    train_df, test_df = split(df)
    train(train_df, test_df)
