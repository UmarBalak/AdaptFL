import os
import logging
import numpy as np
from typing import List, Optional, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from azure.storage.blob import BlobServiceClient, BlobClient
from tensorflow import keras
from datetime import datetime
import tempfile
import io
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import asyncio
import re
import json
import uuid
from fastapi import Body
from dotenv import load_dotenv

load_dotenv()

import os
from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")  # Format: postgresql://user:password@host:port/database
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is missing")

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Client model
class Client(Base):
    __tablename__ = "clients"
    
    csn = Column(String, primary_key=True)
    client_id = Column(String, unique=True, nullable=False)
    api_key = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
SERVER_ACCOUNT_URL = os.getenv("SERVER_ACCOUNT_URL")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")
SERVER_CONTAINER_NAME = os.getenv("SERVER_CONTAINER_NAME")
ARCH_BLOB_NAME = "model_architecture.keras"
CLIENT_NOTIFICATION_URL = os.getenv("CLIENT_NOTIFICATION_URL")

if not CLIENT_ACCOUNT_URL or not SERVER_ACCOUNT_URL:
    logging.error("SAS url environment variable is missing.")
    raise ValueError("Missing required environment variable: SAS url")


try:
    blob_service_client_client = BlobServiceClient(account_url=CLIENT_ACCOUNT_URL)
    blob_service_client_server = BlobServiceClient(account_url=SERVER_ACCOUNT_URL)
except Exception as e:
    logging.error(f"Failed to initialize Azure Blob Service: {e}")
    raise

last_processed_timestamp = 0  # Initialize with 0
latest_version = 0  # Initialize version number


# Store client credentials
# Get the current directory of the script
script_directory = os.path.dirname(os.path.realpath(__file__))
CLIENT_CREDENTIALS_FILE = os.path.join(script_directory, "server_client_credentials.json")
if not os.path.exists(CLIENT_CREDENTIALS_FILE):
    with open(CLIENT_CREDENTIALS_FILE, "w") as f:
        json.dump({}, f)

def load_client_credentials():
    with open(CLIENT_CREDENTIALS_FILE, "r") as f:
        return json.load(f)

def save_client_credentials(credentials):
    with open(CLIENT_CREDENTIALS_FILE, "w") as f:
        json.dump(credentials, f)


def get_model_architecture() -> Optional[object]:
    """
    Load model architecture from blob storage.
    """
    try:
        container_client = blob_service_client_client.get_container_client(CLIENT_CONTAINER_NAME)
        logging.info("Container client initialized successfully.")
        blob_client = container_client.get_blob_client(ARCH_BLOB_NAME)
        logging.info("Blob client initialized successfully.")
        
        # Download architecture file to memory
        arch_data = blob_client.download_blob().readall()
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_file:
            temp_file.write(arch_data)
            temp_path = temp_file.name
        
        model = keras.models.load_model(temp_path)
        os.unlink(temp_path)
        return model
    
    except ImportError as e:
        logging.error(f"Import error while loading model architecture: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None
    except Exception as e:
        logging.error(f"Error loading model architecture: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None


def load_weights_from_blob(
    blob_service_client_client: BlobServiceClient,
    container_name: str,
    model,
    last_processed_timestamp: int
) -> Optional[List[np.ndarray]]:
    try:
        pattern = re.compile(r"client[0-9a-fA-F\-]+_v\d+_(\d{8}_\d{6})\.keras")
        container_client = blob_service_client_client.get_container_client(container_name)

        weights_list = []
        new_last_processed_timestamp = last_processed_timestamp

        blobs = list(container_client.list_blobs())
        # print(blobs)
        for blob in blobs:
            match = pattern.match(blob.name)
            if match:
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


        if not weights_list:
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

        blob_client = blob_service_client_server.get_blob_client(
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
    container_client = blob_service_client_server.get_container_client(SERVER_CONTAINER_NAME)
    blobs = container_client.list_blobs(name_starts_with="g")
    
    latest_version = 0
    for blob in blobs:
        match = re.match(r"g(\d+)\.keras", blob.name)
        if match:
            version = int(match.group(1))
            if version > latest_version:
                latest_version = version
    
    return f"g{latest_version}.keras" if latest_version > 0 else "none"

# Admin authentication function
def verify_admin(api_key: str):
    admin_key = os.getenv("ADMIN_API_KEY", "your_admin_secret_key")
    if api_key != admin_key:
        raise HTTPException(status_code=403, detail="Unauthorized admin access")

# Modified registration endpoint
@app.post("/register")
async def register(
    csn: str = Body(...),
    admin_api_key: str = Body(...),
    db: Session = Depends(get_db)
):
    verify_admin(admin_api_key)
    
    existing_client = db.query(Client).filter(Client.csn == csn).first()
    if existing_client:
        raise HTTPException(status_code=400, detail="Client already registered")
    
    client_id = str(uuid.uuid4())
    api_key = str(uuid.uuid4())
    
    new_client = Client(
        csn=csn,
        client_id=client_id,
        api_key=api_key
    )
    
    db.add(new_client)
    db.commit()
    
    return {
        "status": "success",
        "message": "Client registered successfully",
        "data": {"client_id": client_id, "api_key": api_key}
    }

# Modified WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    db: Session = Depends(get_db)
):
    client = db.query(Client).filter(Client.client_id == client_id).first()
    if not client:
        await websocket.close(code=1008, reason="Unauthorized")
        logging.error(f"Unauthorized connection attempt from client {client_id}")
        return

    await manager.connect(client_id, websocket)
    try:
        latest_model_version = get_latest_model_version()
        await websocket.send_text(f"LATEST_MODEL:{latest_model_version}")
        
        while True:
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
            blob_service_client_client, CLIENT_CONTAINER_NAME, model, last_processed_timestamp
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
    Scheduled task to aggregate weights every minute.
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