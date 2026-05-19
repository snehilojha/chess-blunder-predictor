from fastapi import FastAPI
from routers import predict, health

app = FastAPI(title="Chess Blunder Predictor")

app.include_router(health.router)
app.include_router(predict.router)
