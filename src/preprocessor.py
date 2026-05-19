import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_GAMES_PATH, CLEANED_PATH,
    MIN_ELO, MAX_ELO, TIME_CONTROLS,
    SNAPSHOT_MOVE, BLUNDER_WINDOW, BLUNDER_THRESHOLD, CAP,
)


def _has_missing_evals(evals, start=SNAPSHOT_MOVE, end=SNAPSHOT_MOVE + BLUNDER_WINDOW):
    return any(v != v for v in evals[start:end])


def _cap_evals(evals):
    return np.array([max(-CAP, min(CAP, v)) if v == v else np.nan for v in evals])


def _has_blunder(evals, start=SNAPSHOT_MOVE, end=SNAPSHOT_MOVE + BLUNDER_WINDOW, threshold=BLUNDER_THRESHOLD):
    for offset in [0, 1]:
        side = [min(max(v, -CAP), CAP) for v in evals[start:end][offset::2] if v == v]
        for i in range(1, len(side)):
            if side[i - 1] - side[i] > threshold:
                return 1
    return 0


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time_control"] = df["time_control"].str.split("+").str[0].astype(int)
    df = df[df["time_control"].isin(TIME_CONTROLS)]

    df = df[df["white_elo"].between(MIN_ELO, MAX_ELO) & df["black_elo"].between(MIN_ELO, MAX_ELO)]

    df = df[df["moves"].apply(len) >= SNAPSHOT_MOVE + BLUNDER_WINDOW]

    df = df[~df["evals"].apply(_has_missing_evals)]

    df["evals"] = df["evals"].apply(_cap_evals)

    df["blunder_label"] = df["evals"].apply(_has_blunder)

    return df.reset_index(drop=True)


def run():
    df = pd.read_parquet(RAW_GAMES_PATH)
    cleaned = preprocess(df)
    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(CLEANED_PATH, index=False)
    print(f"Saved {len(cleaned)} games to {CLEANED_PATH}")
    print(f"Blunder rate: {cleaned['blunder_label'].mean():.2%}")


if __name__ == "__main__":
    run()
