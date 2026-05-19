import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pydantic import ValidationError
from routers.predict import PredictRequest
from src.predict import predict


VALID_MOVES = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
               "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5",
               "a4b3", "d7d6", "c2c3"]
VALID_EVALS = [0.1] * 15


# --- PredictRequest validation ---

def test_request_valid():
    req = PredictRequest(
        moves=VALID_MOVES, evals=VALID_EVALS,
        white_elo=1500, black_elo=1500, time_control=600,
    )
    assert req.white_elo == 1500

def test_request_moves_too_short():
    with pytest.raises(ValidationError):
        PredictRequest(
            moves=VALID_MOVES[:10], evals=VALID_EVALS[:10],
            white_elo=1500, black_elo=1500, time_control=600,
        )

def test_request_evals_too_short():
    with pytest.raises(ValidationError):
        PredictRequest(
            moves=VALID_MOVES, evals=VALID_EVALS[:10],
            white_elo=1500, black_elo=1500, time_control=600,
        )

def test_request_length_mismatch():
    with pytest.raises(ValidationError):
        PredictRequest(
            moves=VALID_MOVES, evals=VALID_EVALS + [0.0],
            white_elo=1500, black_elo=1500, time_control=600,
        )

def test_request_elo_out_of_range():
    with pytest.raises(ValidationError):
        PredictRequest(
            moves=VALID_MOVES, evals=VALID_EVALS,
            white_elo=50, black_elo=1500, time_control=600,
        )


# --- predict() ---

def test_predict_returns_both_sides():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.6, 0.4]])

    with patch("src.predict.get_model", return_value=mock_model):
        result = predict(VALID_MOVES, VALID_EVALS, 1500, 1500, 600)

    assert "white_blunder_prob" in result
    assert "black_blunder_prob" in result

def test_predict_probabilities_in_range():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.6, 0.4]])

    with patch("src.predict.get_model", return_value=mock_model):
        result = predict(VALID_MOVES, VALID_EVALS, 1500, 1500, 600)

    assert 0.0 <= result["white_blunder_prob"] <= 1.0
    assert 0.0 <= result["black_blunder_prob"] <= 1.0

def test_predict_uses_model_output():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.25], [0.6, 0.75]])

    with patch("src.predict.get_model", return_value=mock_model):
        result = predict(VALID_MOVES, VALID_EVALS, 1500, 1500, 600)

    assert result["white_blunder_prob"] == 0.25
    assert result["black_blunder_prob"] == 0.75
