import logging
import os
import asyncio
import websockets
import re
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("client.log"),
        logging.StreamHandler()
    ]
)

CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
GLOBAL_MODEL_CONTAINER_NAME = os.getenv("GLOBAL_CONTAINER_NAME")
LOCAL_DOWNLOAD_DIR = os.getenv("LOCAL_DOWNLOAD_DIR")
BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(CONNECTION_STRING)

class WebSocketClient:
    def __init__(self, client_id, server_host="localhost", server_port=8000):
        self.client_id = client_id
        self.server_url = f"ws://{server_host}:{server_port}/ws/{client_id}"
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
            except websockets.exceptions.ConnectionClosed:
                logging.error("WebSocket connection closed unexpectedly")
                await self.handle_disconnection()
            except Exception as e:
                logging.error(f"Connection error: {e}")
                await self.handle_disconnection()

    async def handle_disconnection(self):
        self.connected = False
        logging.info(f"Waiting {self.reconnect_delay} seconds before reconnecting...")
        await asyncio.sleep(self.reconnect_delay)
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    async def handle_messages(self):
        try:
            while True:
                message = await self.websocket.recv()
                logging.info(f"Received message: {message}")
                
                if message.startswith("NEW_MODEL:"):
                    filename = message.split(":")[1]
                    logging.info(f"New global model available: {filename}")
                    await self.update_model(filename)
                
                # Send periodic heartbeat or status updates if needed
                await self.send_status()

        except websockets.exceptions.ConnectionClosed:
            logging.error("Connection closed while handling messages")
            self.connected = False
        except Exception as e:
            logging.error(f"Error handling messages: {e}")
            self.connected = False

    async def send_status(self):
        """Send periodic status updates to server"""
        if self.connected:
            try:
                status_message = f"STATUS:client_{self.client_id}_active"
                await self.websocket.send(status_message)
                await asyncio.sleep(60)  # Send status update every minute
            except Exception as e:
                logging.error(f"Error sending status update: {e}")

    async def update_model(self, filename):
        try:
            local_file_path = await self.download_model(filename)
            if local_file_path:
                logging.info(f"Successfully updated model weights: {local_file_path}")
                return True
        except Exception as e:
            logging.error(f"Error updating model: {e}")
            return False

    async def download_model(self, filename):
        try:
            # Ensure download directory exists
            os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)
            local_file_path = os.path.join(LOCAL_DOWNLOAD_DIR, filename)

            # Download the specific model file
            blob_client = BLOB_SERVICE_CLIENT.get_blob_client(
                container=GLOBAL_MODEL_CONTAINER_NAME, 
                blob=filename
            )

            with open(local_file_path, "wb") as file:
                download_stream = blob_client.download_blob()
                file.write(download_stream.readall())

            logging.info(f"Downloaded model to {local_file_path}")
            return local_file_path

        except Exception as e:
            logging.error(f"Error downloading model: {e}")
            return None

async def run_websocket_service(client_id, host="localhost", port=8000):
    """Independent WebSocket service."""
    while True:
        try:
            ws_client = WebSocketClient(client_id, host, port)
            await ws_client.connect()
        except Exception as e:
            logging.error(f"WebSocket service error: {e}")
            await asyncio.sleep(5)  # Wait before restarting service

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run WebSocket client service')
    parser.add_argument('--client-id', required=True, help='Client ID')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    
    args = parser.parse_args()
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(run_websocket_service(args.client_id, args.host, args.port))
    except KeyboardInterrupt:
        logging.info("Service stopped by user")
    finally:
        loop.close()