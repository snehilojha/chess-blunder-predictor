import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
from src.feature_engineer import side_has_blunder, eval_features, board_features
import chess


# --- side_has_blunder ---

def test_side_has_blunder_detects_white_drop():
    # white evals at even indices: 300, 50 — drop of 250 > threshold
    evals = [0] * 15 + [300, 0, 50, 0]
    assert side_has_blunder(evals, offset=0) == 1

def test_side_has_blunder_detects_black_drop():
    # black evals at odd indices: 300, 50 — drop of 250 > threshold
    evals = [0] * 15 + [0, 300, 0, 50]
    assert side_has_blunder(evals, offset=1) == 1

def test_side_has_blunder_no_blunder():
    evals = [0] * 15 + [100, 100, 100, 100]
    assert side_has_blunder(evals, offset=0) == 0

def test_side_has_blunder_exactly_at_threshold():
    # drop of exactly 200 is not > threshold, so no blunder
    evals = [0] * 15 + [200, 0, 0, 0]
    assert side_has_blunder(evals, offset=0) == 0

def test_side_has_blunder_ignores_nans():
    # NaN at even index (white's slot) — must not shift black's values into white's side
    evals = [0] * 15 + [float("nan"), 0, 300, 0, 50, 0]
    assert side_has_blunder(evals, offset=0) == 1

def test_side_has_blunder_caps_extreme_evals():
    # mate scores get capped to ±1000, drop = 1000 - 50 = 950 > 200
    evals = [0] * 15 + [10000, 0, 50, 0]
    assert side_has_blunder(evals, offset=0) == 1


# --- eval_features ---

def test_eval_features_white_sign():
    evals = [100] * 15
    result = eval_features(evals, offset=0, color=chess.WHITE)
    assert result["eval_at_snapshot"] > 0

def test_eval_features_black_negated():
    # same evals, black perspective should negate
    evals = list(range(15))  # different values at each index
    white = eval_features(evals, offset=0, color=chess.WHITE)
    black_as_white = eval_features(evals, offset=1, color=chess.WHITE)  # no negation
    black = eval_features(evals, offset=1, color=chess.BLACK)           # with negation
    assert black["eval_at_snapshot"] == -black_as_white["eval_at_snapshot"]

def test_eval_features_empty_returns_nans():
    import math
    result = eval_features([], offset=0, color=chess.WHITE)
    assert math.isnan(result["eval_at_snapshot"])
    assert math.isnan(result["eval_volatility"])
    assert math.isnan(result["eval_trend"])

def test_eval_features_volatility_nonnegative():
    evals = [50, 100, 80, 120, 90, 70, 110, 60, 95, 85, 75, 105, 55, 115, 65]
    result = eval_features(evals, offset=0, color=chess.WHITE)
    assert result["eval_volatility"] >= 0


# --- board_features ---

def test_board_features_starting_position():
    # 15 legal opening moves, no captures — material should be balanced
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
             "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5",
             "a4b3", "d7d6", "c2c3"]
    result = board_features(moves, color=chess.WHITE)
    assert result["material_balance"] == 0
    assert result["material_imbalance"] == 0
    assert result["has_castled"] == 1  # e1g1 is in the move list

def test_board_features_no_castle():
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5",
             "b1c3", "g8f6", "d2d3", "d7d6", "c1e3", "c5b6",
             "d1d2", "e8g8", "e1d1"]  # white king moves, not castling
    result = board_features(moves, color=chess.WHITE)
    assert result["has_castled"] == 0

def test_board_features_king_attackers_type():
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
             "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5",
             "a4b3", "d7d6", "c2c3"]
    result = board_features(moves, color=chess.WHITE)
    assert isinstance(result["king_attackers"], int)
    assert result["king_attackers"] >= 0
