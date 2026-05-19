from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.predict import predict

router = APIRouter()


class PredictRequest(BaseModel):
    moves: list[str] = Field(..., min_length=15)
    evals: list[float] = Field(..., min_length=15)
    white_elo: int = Field(..., ge=100, le=3500)
    black_elo: int = Field(..., ge=100, le=3500)
    time_control: int = Field(..., description="Base time control in seconds")

    @model_validator(mode="after")
    def check_lengths_match(self):
        if len(self.moves) != len(self.evals):
            raise ValueError("moves and evals must have the same length")
        return self


class PredictResponse(BaseModel):
    white_blunder_prob: float
    black_blunder_prob: float


@router.post("/predict", response_model=PredictResponse)
def predict_blunder(request: PredictRequest) -> PredictResponse:
    result = predict(
        moves=request.moves,
        evals=request.evals,
        white_elo=request.white_elo,
        black_elo=request.black_elo,
        time_control=request.time_control,
    )
    return PredictResponse(**result)
