import os
import logging
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from azure.storage.blob import BlobServiceClient, BlobClient
from keras.models import load_model
from datetime import datetime
import tempfile
import io
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from contextlib import asynccontextmanager
import asyncio
import requests

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)

# Initialize FastAPI app
app = FastAPI()

# Azure Blob Storage configuration
CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")
SERVER_CONTAINER_NAME = os.getenv("GLOBAL_CONTAINER_NAME")
ARCH_BLOB_NAME = "model_architecture.keras"
CLIENT_NOTIFICATION_URL = os.getenv("CLIENT_NOTIFICATION_URL")

if not all([CONNECTION_STRING, CLIENT_CONTAINER_NAME, SERVER_CONTAINER_NAME]):
    raise ValueError("Missing required environment variables")

try:
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
except Exception as e:
    logging.error(f"Failed to initialize Azure Blob Service: {e}")
    raise

def get_model_architecture() -> Optional[object]:
    """
    Load model architecture from blob storage.
    """
    try:
        container_client = blob_service_client.get_container_client(CLIENT_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(ARCH_BLOB_NAME)
        
        # Download architecture file to memory
        arch_data = blob_client.download_blob().readall()
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_file:
            temp_file.write(arch_data)
            temp_path = temp_file.name
        
        model = load_model(temp_path)
        os.unlink(temp_path)
        return model
    except Exception as e:
        logging.error(f"Error loading model architecture: {e}")
        if 'temp_path' in locals():
            os.unlink(temp_path)
        return None

def load_weights_from_blob(blob_client: BlobClient, model) -> Optional[List[np.ndarray]]:
    """
    Load model weights from a blob.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_file:
            download_stream = blob_client.download_blob()
            temp_file.write(download_stream.readall())
            temp_path = temp_file.name

        model.load_weights(temp_path)
        weights = model.get_weights()
        
        os.unlink(temp_path)

        return weights
    
    except Exception as e:
        logging.error(f"Error loading weights from blob: {e}")
        if 'temp_path' in locals():
            os.unlink(temp_path)
        return None

def save_weights_to_blob(weights: List[np.ndarray], filename: str, model) -> bool:
    """
    Save model weights to a blob.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_file:
            temp_path = temp_file.name
            model.set_weights(weights)
            model.save_weights(temp_path)

        blob_client = blob_service_client.get_blob_client(
            container=SERVER_CONTAINER_NAME, 
            blob=filename
        )
        
        with open(temp_path, "rb") as file:
            blob_client.upload_blob(file, overwrite=True)
        
        logging.info(f"Successfully saved weights to blob: {filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving weights to blob: {e}")
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def federated_averaging(weights_list):
    """
    Perform Federated Averaging on a list of model weights.

    Args:
        weights_list (list): List of model weights from clients.

    Returns:
        avg_weights (list): Federated averaged weights.
    """
    if not weights_list:
        logging.error("No weights available for aggregation")
        return None
    
    avg_weights = []
    for layer_weights in zip(*weights_list):  # Iterate layer-wise
        avg_weights.append(np.mean(layer_weights, axis=0))  # Average weights for each layer
    logging.info("Federated averaging completed successfully.")
    return avg_weights

def notify_client():
    """
    Notify the client to download the latest global model.
    """
    try:
        # client_url = f"{CLIENT_NOTIFICATION_URL}/get_global_model"
        client_url = "http://localhost:8050/get_global_model"
        print(client_url)
        response = requests.get(client_url, verify=False)
        if response.status_code == 200:
            logging.info("Client notified successfully.")
        else:
            logging.error(f"Failed to notify client. Status code: {response.status_code}")
    except Exception as e:
        logging.error(f"Error notifying client: {e}")

@app.get("/aggregate-weights")
async def aggregate_weights():
    """
    Aggregate weights from all client models and save the result.
    """
    try:
        # Load model architecture from blob storage
        model = get_model_architecture()
        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model architecture from blob storage"
            )

        # Get client weights
        container_client = blob_service_client.get_container_client(CLIENT_CONTAINER_NAME)
        blob_list = list(container_client.list_blobs(name_starts_with="client"))

        if not blob_list:
            raise HTTPException(
                status_code=404,
                detail="No client weight files found in the container"
            )

        weights_list = []
        for blob in blob_list:
            if not blob.name.endswith(".keras"):
                continue
                
            blob_client = container_client.get_blob_client(blob)
            weights = load_weights_from_blob(blob_client, model)
            if weights is not None:
                weights_list.append(weights)

        if not weights_list:
            raise HTTPException(
                status_code=404,
                detail="No valid weight files could be loaded"
            )

        # Perform federated averaging
        avg_weights = federated_averaging(weights_list)
        if avg_weights is None:
            raise HTTPException(
                status_code=500,
                detail="Federated averaging failed"
            )

        # Save aggregated weights
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aggregated_weights_{timestamp}.keras"
        if not save_weights_to_blob(avg_weights, filename, model):
            raise HTTPException(
                status_code=500,
                detail="Failed to save aggregated weights"
            )
        # Notify the client
        notify_client()

        return {
            "status": "success",
            "message": f"Aggregated weights saved as {filename}",
            "num_clients": len(weights_list)
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error during aggregation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during aggregation: {str(e)}"
        )

# Scheduler setup
scheduler = BackgroundScheduler()

@scheduler.scheduled_job(CronTrigger(minute="*/1"))
def scheduled_aggregate_weights():
    """
    Scheduled task to aggregate weights every hour.
    """
    logging.info("Scheduled task: Starting weight aggregation process.")
    try:
        asyncio.run(aggregate_weights())
    except Exception as e:
        logging.error(f"Error during scheduled weight aggregation: {e}")

scheduler.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)