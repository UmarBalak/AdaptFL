# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import List
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI()

# Initialize Azure Blob service client
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
if not AZURE_CONNECTION_STRING:
    raise ValueError("Azure connection string not found in .env file")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
CONTAINER_NAME = "model-weights"

# Function to upload file to Azure Blob (reuse from your code)
def upload_file_to_blob(local_file_path: str, blob_name: str):
    """Upload a file to Azure Blob Storage."""
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        return f"Uploaded {local_file_path} to blob {blob_name}"
    except Exception as e:
        return f"Error uploading file to blob: {e}"

# Function to download file from Azure Blob (reuse from your code)
def download_file_from_blob(blob_name: str, local_file_path: str):
    """Download a file from Azure Blob Storage."""
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        with open(local_file_path, "wb") as data:
            data.write(blob_client.download_blob().readall())
        return f"Downloaded blob {blob_name} to {local_file_path}"
    except Exception as e:
        return f"Error downloading file from blob: {e}"

# Route to upload model weights to blob
@app.post("/upload_weights/")
async def upload_weights(file: UploadFile = File(...)):
    """
    Upload model weights file to Azure Blob Storage.
    """
    try:
        local_file_path = f"temp/{file.filename}"
        with open(local_file_path, "wb") as f:
            f.write(file.file.read())
        
        blob_name = file.filename  # Use the original file name for the blob
        upload_message = upload_file_to_blob(local_file_path, blob_name)
        os.remove(local_file_path)  # Clean up the local file after uploading
        return {"message": upload_message}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")

# Route to download model weights from blob
@app.get("/download_weights/{blob_name}")
async def download_weights(blob_name: str):
    """
    Download model weights file from Azure Blob Storage.
    """
    try:
        local_file_path = f"temp/{blob_name}"
        download_message = download_file_from_blob(blob_name, local_file_path)
        return {"message": download_message, "file": local_file_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {e}")
