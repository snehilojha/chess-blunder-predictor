# Chess Blunder Predictor

Binary classifier that predicts whether a chess player will blunder in moves 16–25,
based on game state features extracted at move 15.

**Model:** XGBoost  
**ROC-AUC:** 0.68 (CV: 0.6844 ± 0.0063)  
**Avg Precision:** 0.24 (CV: 0.2585 ± 0.0044)  
**Blunder rate:** ~14.5% (per side)

---

## Problem Definition

A blunder is defined as an eval drop of more than 200 centipawns within moves 16–25,
measured separately for each side. Raw Stockfish evals are interleaved (even indices = white,
odd indices = black), so blunder detection and all eval features operate on per-side slices
before any filtering — this prevents NaN removal from shifting side alignment.

Labels are generated per player per game, giving 2 rows per game (one for white, one for black).
The dataset has 14,541 games → 29,082 training rows.

---

## Project Structure

```
chess_blunder/
├── data/
│   ├── raw/                        ← source PGN file
│   └── processed/
│       ├── games.parquet           ← parsed games (output of data_loader)
│       ├── cleaned_games.parquet   ← filtered games (output of preprocessor)
│       └── features.parquet        ← feature matrix (output of feature_engineer)
├── models/
│   ├── preprocessing_metadata.json
│   └── mlflow_artifacts/           ← MLflow model artifacts
├── notebooks/
│   ├── 01_data_loader.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── feature_engineer.py         ← blunder labels + feature extraction
│   ├── train.py                    ← CV, final training, MLflow logging
│   ├── evaluate.py                 ← metrics dict + SHAP importance
│   └── predict.py                  ← lazy model loader + inference
├── routers/
│   ├── predict.py                  ← POST /predict (Pydantic validation)
│   └── health.py                   ← GET /health (model load check)
├── tests/
│   ├── test_feature.py             ← 13 unit tests for feature_engineer
│   └── test_predict.py             ← 8 unit tests for predict + router validation
├── config.py                       ← all paths, constants, hyperparameters
├── main.py                         ← FastAPI entry point
├── mlruns.db                       ← MLflow tracking store (SQLite)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Features

All features are extracted at move 15 (snapshot), looking back at moves 1–15.

| Feature | Description |
|---|---|
| `player_elo` | Player's Elo rating |
| `elo_diff` | Player Elo minus opponent Elo |
| `time_control` | Base time in seconds (600 / 900 / 1800) |
| `eval_at_snapshot` | Stockfish eval at move 15 from player's perspective |
| `eval_volatility` | Std dev of player's evals over moves 1–15 |
| `eval_trend` | Slope of player's last 5 evals (positive = improving) |
| `material_balance` | Player material minus opponent material (centipawns) |
| `material_imbalance` | Absolute difference in material |
| `king_attackers` | Number of opponent pieces attacking the player's king at move 15 |
| `has_castled` | Whether the player has castled by move 15 |

Evals are capped at ±1000 to suppress mate score sentinels (±10000 from Stockfish).
Black's directional features (`eval_at_snapshot`, `eval_trend`) are negated since raw
Stockfish evals are always from white's absolute perspective.

---

## Data

**Source:** [Lichess Open Database](https://database.lichess.org) — June 2017 PGN with embedded Stockfish evals.

**Filters applied:**
- Elo: 1000–2500 for both players
- Time controls: 600s, 900s, 1800s (10, 15, 30 min)
- Minimum game length: 25 moves
- Capped at 1000 games per (white_elo_bracket, black_elo_bracket, time_control) to balance the dataset

---

## Modeling

**Algorithm:** XGBoost (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`)

**Class imbalance:** Handled via `scale_pos_weight=5.82` (ratio of negatives to positives at ~14.5% blunder rate).

**Train/test split:** By `game_id` (not rows) to prevent white and black rows from the same
game appearing in both splits.

**Cross-validation:** 5-fold stratified CV also split by `game_id` for the same reason.
Game-level label for stratification uses `.max()` — a game is positive if either the white
or black row is a blunder.

---

## Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## Run with Docker

```bash
docker compose up
```

API at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

> **Note:** `mlruns.db` and `models/` are mounted as volumes — retraining updates the model
> registry without rebuilding the image.

---

## API

### `GET /health`

Returns API status, model name, version, and whether the model loaded successfully.

```json
{
  "status": "ok",
  "model": "blunder_predictor",
  "version": 1,
  "model_status": "loaded"
}
```

### `POST /predict`

Predicts blunder probability for both sides given game state at move 15.

**Validation:**
- `moves` and `evals` must each have at least 15 elements
- `moves` and `evals` must be the same length
- Elo must be between 100 and 3500

**Request:**
```json
{
  "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6",
            "b5a4", "g8f6", "e1g1", "f8e7", "f1e1", "b7b5",
            "a4b3", "d7d6", "c2c3"],
  "evals": [0.3, -0.3, 0.4, -0.2, 0.5, -0.1, 0.6, -0.3,
            0.4, -0.2, 0.5, -0.4, 0.3, -0.1, 0.4],
  "white_elo": 1500,
  "black_elo": 1500,
  "time_control": 600
}
```

**Response:**
```json
{
  "white_blunder_prob": 0.1052,
  "black_blunder_prob": 0.1127
}
```

**curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "moves": ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6","e1g1","f8e7","f1e1","b7b5","a4b3","d7d6","c2c3"],
    "evals": [0.3,-0.3,0.4,-0.2,0.5,-0.1,0.6,-0.3,0.4,-0.2,0.5,-0.4,0.3,-0.1,0.4],
    "white_elo": 1500,
    "black_elo": 1500,
    "time_control": 600
  }'
```

---

## Tests

```bash
pytest tests/ -v
```

21 tests across feature engineering logic and API validation.
