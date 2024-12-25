import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import argparse

# Load environment variables from .env file
load_dotenv()

# Now retrieve the connection string
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")

# Rest of your code remains the same...
if not AZURE_CONNECTION_STRING:
    raise ValueError("Azure connection string not found in .env file")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

def upload_weights(local_file_path, blob_name):
    """
    Upload a local file to Azure Blob Storage.
    
    Args:
        local_file_path (str): Path to the local file to upload.
        blob_name (str): Name to use for the file in the blob container.
    """
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f"Uploaded {local_file_path} to Azure Blob Storage as {blob_name}.")
    except Exception as e:
        print(f"Error uploading weights: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Upload weights to Azure Blob Storage.")
    parser.add_argument("local_file_path", type=str, help="Path to the local weights file (e.g., 'weights.npy').")
    parser.add_argument("blob_name", type=str, help="Name for the file in the blob container (e.g., 'client1_weights.npy').")
    args = parser.parse_args()

    # Ensure the Azure connection string is set
    if not AZURE_CONNECTION_STRING:
        raise ValueError("AZURE_CONNECTION_STRING environment variable is not set.")

    # Call the upload function
    upload_weights(args.local_file_path, args.blob_name)
