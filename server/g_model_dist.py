import os
import shutil
import logging
import numpy as np
import gzip
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

def setup_logger():
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        filename="global_server/logs/global_model_distribution.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def distribute_global_model(processed_blob_name, client_dirs):
    """
    Distribute the global model to all clients.

    Args:
        processed_blob_name (str): Name of the processed weights blob.
        client_dirs (list): List of client directories.
    """
    load_dotenv()
    AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client("model-weights")

    try:
        for client_dir in client_dirs:
            try:
                client_model_path = os.path.join(client_dir, "models", "global", os.path.basename(processed_blob_name))
                os.makedirs(os.path.dirname(client_model_path), exist_ok=True)
                
                # Download the processed weights from Azure Blob Storage
                blob_client = container_client.get_blob_client(processed_blob_name)
                with open(client_model_path, "wb") as f:
                    f.write(blob_client.download_blob().readall())
                
                logging.info(f"Distributed global model to {client_model_path}")
            except Exception as e:
                logging.error(f"Error distributing model to {client_dir}: {e}")
    except Exception as e:
        logging.error(f"Global model distribution failed: {e}")

def main():
    # Setup logging
    setup_logger()

    # Paths and directories
    client_dirs = [
        "../AdaptFL_Project/client1",
        "../AdaptFL_Project/client2",
        "../AdaptFL_Project/client3"
    ]

    # Processed weights blob name
    processed_blob_name = "processed_weights/processed_weights_latest.h5"

    # Distribute the model to clients
    distribute_global_model(processed_blob_name, client_dirs)

if __name__ == "__main__":
    main()