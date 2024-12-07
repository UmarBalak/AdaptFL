from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import numpy as np
from keras.models import load_model
import logging
from dotenv import load_dotenv

app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Now retrieve the connection string
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")

# Rest of your code remains the same...
if not AZURE_CONNECTION_STRING:
    raise ValueError("Azure connection string not found in .env file")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
BLOB_CONTAINER_NAME = "model-weights"


blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

# Setup logger for server
logging.basicConfig(level=logging.INFO)

def save_model_to_blob(client_id, model_file):
    """Upload the model to Azure Blob Storage."""
    blob_client = container_client.get_blob_client(f"{client_id}_model.h5")
    with open(model_file, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    logging.info(f"Model for {client_id} uploaded to Azure Blob Storage.")

def download_model_from_blob(client_id):
    """Download the model from Azure Blob Storage."""
    blob_client = container_client.get_blob_client(f"{client_id}_model.h5")
    with open(f"{client_id}_model.h5", "wb") as file:
        blob_client.download_blob().readinto(file)
    logging.info(f"Model for {client_id} downloaded from Azure Blob Storage.")

@app.post("/upload_weights")
async def upload_weights(client_id: str, file: UploadFile):
    """Upload client weights to server."""
    try:
        weights_path = f"./weights/{client_id}_weights.npy"
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)

        with open(weights_path, "wb") as f:
            f.write(await file.read())
        
        logging.info(f"Weights from {client_id} uploaded successfully.")
        return {"message": "Weights uploaded successfully."}

    except Exception as e:
        logging.error(f"Failed to upload weights for {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Error uploading weights")

@app.post("/aggregate_weights")
def aggregate_weights():
    """Aggregate client weights and update the global model."""
    try:
        weights_list = []
        for file in os.listdir("./weights"):
            if file.endswith(".npy"):
                weights_list.append(np.load(os.path.join("./weights", file), allow_pickle=True))
        
        # Federated Averaging
        avg_weights = [np.mean(np.array(weights), axis=0) for weights in zip(*weights_list)]

        # Load global model, update weights and save back to blob storage
        global_model = load_model("global_model.h5")
        global_model.set_weights(avg_weights)
        
        global_model.save("global_model.h5")
        save_model_to_blob("global_server", "global_model.h5")

        return {"message": "Weights aggregated and global model updated."}

    except Exception as e:
        logging.error(f"Error during weight aggregation: {e}")
        raise HTTPException(status_code=500, detail="Error aggregating weights")

@app.get("/download_weights")
def download_weights():
    """Download the global model weights."""
    try:
        download_model_from_blob("global_server")
        return {"message": "Global model downloaded."}

    except Exception as e:
        logging.error(f"Failed to download global model: {e}")
        raise HTTPException(status_code=500, detail="Error downloading global model")

@app.get("/")
def root():
    return {"message": "Welcome to the AdaptFL Global Server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
