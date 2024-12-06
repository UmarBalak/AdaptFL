from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from keras.models import load_model
import os
import json
import numpy as np
import logging

app = FastAPI()


def save_global_model_and_weights(global_model, global_model_path):
    """
    Save the global model and its weights.

    Args:
        global_model (keras.Model): The aggregated global model.
        global_model_path (str): Path to save the global model and weights.
    """
    # Save the global model in .h5 format
    global_model.save(global_model_path)
    print(f"Global model saved at {global_model_path}")

    # Save the weights in .npy format
    weights_dir = os.path.dirname(global_model_path)
    weights_path = os.path.join(weights_dir, "global_weights.npy")
    np.save(weights_path, global_model.get_weights())
    print(f"Global weights saved at {weights_path}")


# Paths
GLOBAL_MODEL_PATH = "../AdaptFL_Project/global_server/global_model/global_model.h5"
WEIGHTS_DIR = "../AdaptFL_Project/global_server/weights"

# Ensure directories exist
os.makedirs(WEIGHTS_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Welcome to the AdaptFL Global Server"}

@app.get("/get_global_model")
def get_global_model():
    """Serve the global model file to clients."""
    if not os.path.exists(GLOBAL_MODEL_PATH):
        raise HTTPException(status_code=404, detail="Global model not found")
    return JSONResponse(content={"model_path": GLOBAL_MODEL_PATH})

@app.post("/upload_weights")
async def upload_weights(client_id: str, file: UploadFile):
    """
    Endpoint for clients to upload model weights.
    Args:
        client_id (str): Unique identifier for the client uploading weights.
        file (UploadFile): Uploaded weights file in `.npy` format.
    """
    try:
        # Define the path where the weights will be saved
        weights_path = os.path.join(WEIGHTS_DIR, f"{client_id}_weights.npy")
        
        # Ensure that the file is of correct format (e.g., .npy)
        if not file.filename.endswith(".npy"):
            raise HTTPException(status_code=400, detail="Invalid file format. Only .npy files are accepted.")
        
        # Save the uploaded weights file to the specified path
        with open(weights_path, "wb") as f:
            f.write(await file.read())
        
        return {"message": f"Weights received from {client_id} and saved successfully."}
    
    except Exception as e:
        # Log and raise an exception if there is any error during the process
        raise HTTPException(status_code=500, detail=f"Error saving weights: {str(e)}")

# @app.post("/aggregate_weights")
# def aggregate_weights(WEIGHTS_DIR, GLOBAL_MODEL_PATH):
    try:
        # Load the global model (the architecture and weights)
        global_model = load_model(GLOBAL_MODEL_PATH)
        global_weights = global_model.get_weights()

        # Gather all client models and their weights
        client_weights = []
        for weight_file in os.listdir(WEIGHTS_DIR):
            weight_path = os.path.join(WEIGHTS_DIR, weight_file)
            # Load client model and get its weights
            client_model = load_model(weight_path)
            client_weights.append(client_model.get_weights())

        # Federated Averaging: Compute the average weights
        avg_weights = []
        for layer_weights in zip(*client_weights):
            avg_weights.append(np.mean(layer_weights, axis=0))

        # Set the averaged weights to the global model
        global_model.set_weights(avg_weights)

        # Save the updated global model
        global_model.save(GLOBAL_MODEL_PATH)
        logging.info(f"Global model updated and saved at {GLOBAL_MODEL_PATH}")

    except Exception as e:
        logging.error(f"Error during aggregation: {str(e)}")

@app.post("/aggregate_weights")
def aggregate_weights():
    """
    Aggregate weights from all clients using Federated Averaging.
    This updates the global model.
    """
    try:
        # Load global model
        if not os.path.exists(GLOBAL_MODEL_PATH):
            raise HTTPException(status_code=404, detail="Global model not found")
        
        global_model = load_model(GLOBAL_MODEL_PATH)

        # Gather all client weights
        client_weights = []
        for weight_file in os.listdir(WEIGHTS_DIR):
            weight_path = os.path.join(WEIGHTS_DIR, weight_file)
            client_weights.append(np.load(weight_path, allow_pickle=True))

        # Federated Averaging
        avg_weights = [
            np.mean(layer_weights, axis=0)
            for layer_weights in zip(*client_weights)
        ]

        # Update the global model with averaged weights
        global_model.set_weights(avg_weights)

        # Save the global model and weights
        save_global_model_and_weights(global_model, GLOBAL_MODEL_PATH)

        return {"message": "Global model updated with aggregated weights"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during aggregation: {str(e)}")


from fastapi.responses import FileResponse

@app.get("/download_weights")
def download_weights():
    """
    Endpoint for clients to download the global weights.
    """
    weights_path = os.path.join(WEIGHTS_DIR, "global_weights.npy")
    if not os.path.exists(weights_path):
        raise HTTPException(status_code=404, detail="Global weights not found")
    return FileResponse(weights_path, media_type='application/octet-stream', filename="global_weights.npy")
