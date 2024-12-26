from fastapi import FastAPI, HTTPException
from azure.storage.blob import BlobServiceClient
import os
from datetime import datetime
import re
from dotenv import load_dotenv
load_dotenv() 

CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
GLOBAL_MODEL_CONTAINER_NAME = os.getenv("GLOBAL_CONTAINER_NAME")

blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service_client.get_container_client(container=GLOBAL_MODEL_CONTAINER_NAME)

# Regex pattern to match filenames with the exact timestamp format (YYYYMMDD_HHMMSS)
pattern = re.compile(r"aggregated_weights_(\d{8}_\d{6})\.keras")

app = FastAPI()

@app.get("/")
async def root():
    return {"Hello World": "Welcome to the Global Model Server!"}

@app.get("/get_global_model")
async def get_global_model():
    latest_blob = None
    latest_timestamp = 0

    # Iterate through blobs in the container
    for blob in container_client.list_blobs():
        match = pattern.match(blob.name)
        if match:
            timestamp = int(match.group(1))  # Extract the timestamp
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_blob = blob.name

    # Ensure we found a file
    if latest_blob:
        timestamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Latest file: {latest_blob}")
        # Download the latest file
        blob_client = blob_service_client.get_blob_client(container=GLOBAL_MODEL_CONTAINER_NAME, blob=latest_blob)
        local_file_path = f"D:\AdaptFL\client\model_weights\global\latest_global_weights{timestamp2}.keras"
        with open(local_file_path, "wb") as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())
        print(f"Downloaded latest weights to {local_file_path}")
    else:
        print("No matching files found in the container.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)