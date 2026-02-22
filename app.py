
from fastapi import FastAPI
from api.routers import root, predict

app = FastAPI(
    title="FlightOnTime API",
    description="Not using IA now, right Everton?",
    version="0.1.0"
)

app.include_router(root.router)
app.include_router(predict.router)

