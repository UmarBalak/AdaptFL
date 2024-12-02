from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import json

# Correct path to configuration
CONFIG_PATH = "global_server/config.json"

# Load configuration
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Directory to save model updates
MODEL_SAVE_PATH = "AdaptFL/models/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# FastAPI app
app = FastAPI()

# Initialize global model weights
global_weights = None

# Pydantic model for request validation
class WeightsRequest(BaseModel):
    weights: list

# Endpoint: Receive client weights
@app.post("/send_weights/")
async def receive_weights(data: WeightsRequest):
    global global_weights
    client_weights = np.array(data.weights)

    # Aggregate weights
    if global_weights is None:
        global_weights = client_weights
    else:
        global_weights = (global_weights + client_weights) / 2  # Simple averaging

    return {"message": "Weights received and aggregated successfully."}

# Endpoint: Send global weights to clients
@app.get("/get_weights/")
async def send_weights():
    if global_weights is None:
        raise HTTPException(status_code=400, detail="Global weights not initialized.")
    return {"weights": global_weights.tolist()}
