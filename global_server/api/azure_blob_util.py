import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError

# Load environment variables from .env file
load_dotenv()

# Now retrieve the connection string
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")

# Rest of your code remains the same...
if not AZURE_CONNECTION_STRING:
    raise ValueError("Azure connection string not found in .env file")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
CONTAINER_NAME = "model-weights"

def upload_file_to_blob(local_file_path, blob_name):
    """
    Upload a file to Azure Blob Storage.
    Args:
        local_file_path (str): Path to the local file.
        blob_name (str): Name for the file in the blob container.
    """
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f"Uploaded {local_file_path} to blob {blob_name}")
    except FileNotFoundError:
        print(f"Error: File {local_file_path} not found")
    except ResourceNotFoundError:
        print(f"Error: Container {CONTAINER_NAME} not found")
    except Exception as e:
        print(f"Error uploading file to blob: {e}")

def download_file_from_blob(blob_name, local_file_path):
    """
    Download a file from Azure Blob Storage.
    Args:
        blob_name (str): Name of the file in the blob container.
        local_file_path (str): Path to save the downloaded file locally.
    """
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

        with open(local_file_path, "wb") as data:
            data.write(blob_client.download_blob().readall())

        print(f"Downloaded blob {blob_name} to {local_file_path}")
    except ResourceNotFoundError:
        print(f"Error: Blob {blob_name} not found")
    except Exception as e:
        print(f"Error downloading file from blob: {e}")

def list_blobs_in_container():
    """
    List all blobs in the container.
    """
    try:
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blobs = container_client.list_blobs()

        print("Blobs in container:")
        for blob in blobs:
            print(f" - {blob.name}")
    except ResourceNotFoundError:
        print(f"Error: Container {CONTAINER_NAME} not found")
    except Exception as e:
        print(f"Error listing blobs: {e}")