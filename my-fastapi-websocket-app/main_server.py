import os
import logging
import numpy as np
from typing import List, Optional, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from azure.storage.blob import BlobServiceClient, BlobClient
from keras.models import load_model
from datetime import datetime
import tempfile
import io
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import asyncio
import re
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

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):

        await websocket.accept()
        self.active_connections[client_id] = websocket
        logging.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logging.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast_model_update(self, message: str):
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
                logging.info(f"Update notification sent to client {client_id}")
            except Exception as e:
                logging.error(f"Failed to send update to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)

# Initialize FastAPI app with connection manager
manager = ConnectionManager()
app = FastAPI()

# Azure Blob Storage configuration
CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")
SERVER_CONTAINER_NAME = os.getenv("GLOBAL_CONTAINER_NAME")
ARCH_BLOB_NAME = "model_architecture.keras"
CLIENT_NOTIFICATION_URL = os.getenv("CLIENT_NOTIFICATION_URL")

if not CONNECTION_STRING:
    logging.error("AZURE_CONNECTION_STRING environment variable is missing.")
    raise ValueError("Missing required environment variable: AZURE_CONNECTION_STRING")


try:
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
except Exception as e:
    logging.error(f"Failed to initialize Azure Blob Service: {e}")
    raise

last_processed_timestamp = 0  # Initialize with 0
latest_version = 0  # Initialize version number

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

def load_weights_from_blob(
    blob_service_client: BlobServiceClient,
    container_name: str,
    model,
    last_processed_timestamp: int
) -> Optional[List[np.ndarray]]:
    try:
        pattern = re.compile(r"local_weights_client\d+_v\d+_(\d{8}_\d{6})\.keras")
        container_client = blob_service_client.get_container_client(container_name)

        weights_list = []
        new_last_processed_timestamp = last_processed_timestamp
        # debugging
        weights_files = []
        weights_files_to_aggregate = []
        present_files_in_blob = []
        ############
        for blob in container_client.list_blobs():
            match = pattern.match(blob.name)
            if match:
                present_files_in_blob.append(blob.name)
                timestamp_str = match.group(1)
                timestamp_int = int(timestamp_str.replace("_", ""))
                if timestamp_int > last_processed_timestamp:
                    blob_client = container_client.get_blob_client(blob.name)
                    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_file:
                        download_stream = blob_client.download_blob()
                        temp_file.write(download_stream.readall())
                        temp_path = temp_file.name
                    model.load_weights(temp_path)
                    weights = model.get_weights()
                    os.unlink(temp_path)

                    weights_list.append(weights)
                    new_last_processed_timestamp = max(new_last_processed_timestamp, timestamp_int)

                    # debugging
                    weights_files.append((timestamp_int, blob.name))
                    weights_files_to_aggregate.append(blob.name)
                    #########

        # debugging
        for i in range(5):
            print("******************************************************************************************************************************************************************************")
        print(f"Present files in blob: {present_files_in_blob}", len(present_files_in_blob))
        print(f"Files to aggregate: {weights_files_to_aggregate}", len(weights_files_to_aggregate))
        print("--------------------------------------------------------")
        ###########

        if not weights_files:
            logging.info(f"No new weights found since {last_processed_timestamp}.")
            return None, last_processed_timestamp
        
        logging.info(f"Loaded weights from {len(weights_list)} files.")
        return weights_list, new_last_processed_timestamp

    except Exception as e:
        logging.error(f"Error loading weights: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None, last_processed_timestamp

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

def get_versioned_filename(version: int, prefix="g", extension="keras"):
    filename = f"{prefix}{version}.{extension}"
    return filename

def get_latest_model_version() -> str:
    # Fetch the latest model version based on saved files or in-memory state
    container_client = blob_service_client.get_container_client(SERVER_CONTAINER_NAME)
    blobs = container_client.list_blobs(name_starts_with="g")
    
    latest_version = 0
    for blob in blobs:
        match = re.match(r"g(\d+)\.keras", blob.name)
        if match:
            version = int(match.group(1))
            if version > latest_version:
                latest_version = version
    
    return f"g{latest_version}.keras" if latest_version > 0 else "none"


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(client_id, websocket)
    try:
        # On connection, send the latest model version to the client
        latest_model_version = get_latest_model_version()  # Retrieve the latest model version from server
        await websocket.send_text(f"LATEST_MODEL:{latest_model_version}")
        
        while True:
            # Handle messages from the client (optional heartbeat or other logic)
            data = await websocket.receive_text()
            logging.info(f"Message from client {client_id}: {data}")
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logging.error(f"Error in websocket connection for client {client_id}: {e}")
        await manager.disconnect(client_id)


@app.get("/aggregate-weights")
async def aggregate_weights():
    """
    Aggregate weights from all client models and save the result.
    """
    global last_processed_timestamp  # Use the global variable to track the last processed timestamp
    global latest_version  # Use the global variable to track the latest version

    try:
        # Load model architecture from blob storage
        model = get_model_architecture()
        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to load model architecture from blob storage"
            )

        # Get client weights with timestamp filtering
        weights_list, new_last_processed_timestamp = load_weights_from_blob(
            blob_service_client, CLIENT_CONTAINER_NAME, model, last_processed_timestamp
        )

        if not weights_list:
            logging.info("No valid weight files could be loaded for aggregation.")
            return {
                "status": "no_update",
                "message": "No new weights found for aggregation.",
                "num_clients": 0
            }

        # Perform federated averaging
        avg_weights = federated_averaging(weights_list)
        if avg_weights is None:
            raise HTTPException(
                status_code=500,
                detail="Federated averaging failed"
            )

        # Save aggregated weights
        latest_version += 1
        filename = get_versioned_filename(latest_version)
        if save_weights_to_blob(avg_weights, filename, model):
            # Update the last processed timestamp only after successful save
            last_processed_timestamp = new_last_processed_timestamp

            # Notify clients only when new weights are saved
            await manager.broadcast_model_update(f"NEW_MODEL:{filename}")
            logging.info(f"Aggregation complete, clients notified about {filename}")
        else:
            logging.error("Failed to save aggregated weights")

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