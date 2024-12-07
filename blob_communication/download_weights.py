import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import argparse

# Load environment variables from .env file
load_dotenv()

# Now retrieve the connection string
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")

# Rest of your code remains the same...
if not AZURE_CONNECTION_STRING:
    raise ValueError("Azure connection string not found in .env file")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
CONTAINER_NAME = "model-weights"

def download_weights(blob_name, local_file_path):
    """
    Download a file from Azure Blob Storage.
    
    Args:
        blob_name (str): Name of the file in the blob container (e.g., 'client1_weights.npy').
        local_file_path (str): Path to save the downloaded file locally (e.g., './weights.npy').
    """
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

        with open(local_file_path, "wb") as data:
            data.write(blob_client.download_blob().readall())

        print(f"Downloaded {blob_name} from Azure Blob Storage to {local_file_path}.")
    except Exception as e:
        print(f"Error downloading weights: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download weights from Azure Blob Storage.")
    parser.add_argument("blob_name", type=str, help="Name of the file in the blob container (e.g., 'client1_weights.npy').")
    parser.add_argument("local_file_path", type=str, help="Path to save the downloaded file (e.g., './weights.npy').")
    args = parser.parse_args()

    # Ensure the Azure connection string is set
    if not AZURE_CONNECTION_STRING:
        raise ValueError("AZURE_CONNECTION_STRING environment variable is not set.")

    # Call the download function
    download_weights(args.blob_name, args.local_file_path)
