import logging
import os
import time
import glob
import asyncio
import websockets
import requests
from model_architecture import build_model
from model_training import main as train_main
from preprocessing import preprocess_client_data
from azure.storage.blob import BlobServiceClient
from datetime import datetime
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
GLOBAL_MODEL_CONTAINER_NAME = os.getenv("GLOBAL_CONTAINER_NAME")
print(GLOBAL_MODEL_CONTAINER_NAME)
LOCAL_DOWNLOAD_DIR = os.getenv("LOCAL_DOWNLOAD_DIR")
BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(CONNECTION_STRING)

class WebSocketClient:
    def __init__(self, client_id, server_url):
        self.client_id = client_id
        self.server_url = f"{server_url}/ws/{client_id}"
        self.websocket = None
        self.connected = False
        self.reconnect_delay = 5  # Initial reconnect delay in seconds
        self.max_reconnect_delay = 300  # Maximum reconnect delay (5 minutes)

    async def connect(self):
        while True:
            try:
                self.websocket = await websockets.connect(self.server_url)
                self.connected = True
                self.reconnect_delay = 5  # Reset delay on successful connection
                logging.info(f"Connected to server: {self.server_url}")
                await self.handle_messages()
            except Exception as e:
                self.connected = False
                logging.error(f"Connection error: {e}")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    async def handle_messages(self):
        try:
            while True:
                message = await self.websocket.recv()
                if message.startswith("NEW_MODEL:"):
                    filename = message.split(":")[1]
                    logging.info(f"New global model available: {filename}")
                    await self.update_model()
        except Exception as e:
            logging.error(f"Error handling messages: {e}")
            self.connected = False

    async def update_model(self):
        try:
            latest_blob, local_file_path, latest_timestamp = self.download_latest_model()
            logging.info(f"Successfully updated model weights: {local_file_path}")
            return True
        except Exception as e:
            logging.error(f"Error updating model: {e}")
            return False

    def download_latest_model(self):
        container_client = BLOB_SERVICE_CLIENT.get_container_client(GLOBAL_MODEL_CONTAINER_NAME)

        latest_blob = None
        latest_timestamp = None
        pattern = re.compile(r"global_weights_(\d{8}_\d{6})\.keras")

        for blob in container_client.list_blobs():
            logging.info(f"Found blob: {blob.name}")
            match = pattern.match(blob.name)
            if match:
                timestamp = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
                if not latest_timestamp or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_blob = blob.name
        print(latest_blob)
        if not latest_blob:
            raise Exception("No matching files found in the container.")

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)
        local_file_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"latest_global_weights_{timestamp_str}.keras")

        blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=GLOBAL_MODEL_CONTAINER_NAME, blob=latest_blob)
        with open(local_file_path, "wb") as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())

        logging.info(f"Downloaded latest weights to {local_file_path}")

        return latest_blob, local_file_path, latest_timestamp

def setup_logger(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "client.log"),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def preprocess_data(client_id, raw_data_path, preprocessed_data_path, image_size=(128, 128)):
    """Preprocess raw data for the client."""
    try:
        preprocess_client_data(raw_data_path, preprocessed_data_path, client_id, image_size)
        logging.info(f"Preprocessing completed successfully for {client_id}")
    except Exception as e:
        logging.error(f"Error in preprocessing for {client_id}: {e}")

def train(client_id, data_path, save_dir, epochs=1, batch_size=32):
    """Train the model for the client."""
    try:
        train_main(client_id, data_path, save_dir, build_model, epochs, batch_size)
        logging.info(f"Training completed successfully for {client_id}")
    except Exception as e:
        logging.error(f"Error during training for {client_id}: {e}")

def check_raw_data(raw_data_dir):
    """Check for .hdf5 files in raw data directory."""
    hdf5_files = glob.glob(os.path.join(raw_data_dir, "*.hdf5"))
    return hdf5_files[0] if hdf5_files else None

async def run_data_service(client_id):
    """Independent data processing service."""
    raw_data_dir = f"D:/AdaptFL/client{client_id}/data"
    preprocessed_data_path = f"D:/AdaptFL/client{client_id}/preprocessed_data"
    save_dir = f"D:/AdaptFL"
    log_dir = f"D:/AdaptFL/client{client_id}/logs"

    setup_logger(log_dir)

    while True:
        try:
            hdf5_file = check_raw_data(raw_data_dir)
            
            if hdf5_file:
                logging.info(f"Found new data: {hdf5_file}")
                print(f"Found new data: {hdf5_file}")
                try:
                    preprocess_data(client_id, hdf5_file, preprocessed_data_path)
                    train(client_id, preprocessed_data_path, save_dir)
                    os.remove(hdf5_file)
                    logging.info("Processed file deleted, waiting 6 hours")
                    print("Processed file deleted, waiting 2 minutes")
                    await asyncio.sleep(240)
                except Exception as e:
                    logging.error(f"Error processing data: {e}")
                    print(f"Error processing data: {e}")
                    await asyncio.sleep(60)
            else:
                logging.info("No new data found, checking again in 1 hour")
                print("No new data found, checking again in 1 minute")
                await asyncio.sleep(60)
        except Exception as e:
            logging.error(f"Data service error: {e}")
            await asyncio.sleep(60)

async def run_websocket_service(client_id):
    """Independent WebSocket service."""
    while True:
        try:
            ws_client = WebSocketClient(client_id, "ws://localhost:8000")
            await ws_client.connect()
        except Exception as e:
            logging.error(f"WebSocket service error: {e}")
            await asyncio.sleep(5)  # Wait before restarting service


if __name__ == "__main__":

    client_id = "1"

    loop = asyncio.get_event_loop()

    # Create the independent tasks
    websocket_task = loop.create_task(run_websocket_service(client_id))
    data_task = loop.create_task(run_data_service(client_id))

    try:
        # Run both services indefinitely
        loop.run_forever()
    except KeyboardInterrupt:
        # Handle graceful shutdown
        websocket_task.cancel()
        data_task.cancel()
        loop.run_until_complete(asyncio.gather(websocket_task, data_task, return_exceptions=True))
    finally:
        loop.close()