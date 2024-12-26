from fastapi import FastAPI, HTTPException
from azure.storage.blob import BlobServiceClient
import os
from datetime import datetime
import re
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
GLOBAL_MODEL_CONTAINER_NAME = os.getenv("GLOBAL_CONTAINER_NAME")
LOCAL_DOWNLOAD_DIR = os.getenv("LOCAL_DOWNLOAD_DIR", "D:/AdaptFL/client/model_weights/global/")

# Azure Blob Service client
try:
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
except Exception as e:
    logging.error(f"Failed to connect to Azure Blob Service: {e}")
    raise

container_client = blob_service_client.get_container_client(container=GLOBAL_MODEL_CONTAINER_NAME)

# Regex pattern for filenames
pattern = re.compile(r"aggregated_weights_(\d{8}_\d{6})\.keras")

# Logging setup
logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/")
async def root():
    return {"Hello World": "Welcome to the Global Model Server!"}

@app.get("/get_global_model")
async def get_global_model():
    latest_blob = None
    latest_timestamp = None

    # Iterate through blobs in the container
    for blob in container_client.list_blobs():
        match = pattern.match(blob.name)
        if match:
            timestamp = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            if not latest_timestamp or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_blob = blob.name

    # Ensure we found a file
    if not latest_blob:
        raise HTTPException(status_code=404, detail="No matching files found in the container.")

    # Create download path
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)
    local_file_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"latest_global_weights_{timestamp_str}.keras")

    try:
        # Download the latest file
        blob_client = blob_service_client.get_blob_client(container=GLOBAL_MODEL_CONTAINER_NAME, blob=latest_blob)
        with open(local_file_path, "wb") as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())

        logging.info(f"Downloaded latest weights to {local_file_path}")

        # Return success response
        return {
            "status": "success",
            "file_name": latest_blob,
            "local_path": local_file_path,
            "timestamp": latest_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        logging.error(f"Error downloading the file: {e}")
        raise HTTPException(status_code=500, detail="Error downloading the global model.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)