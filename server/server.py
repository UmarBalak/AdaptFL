from fastapi import FastAPI, HTTPException, UploadFile, File
from azure.storage.blob import BlobServiceClient
import os

# Initialize FastAPI app
app = FastAPI()

# Azure Blob Storage configuration
CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CONTAINER_NAME = "client-weights"
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

@app.get("/")
async def root():
    return {"Message": "Welcome to AdaptFL server."}

@app.get("/files/{filename}")
async def download_file(filename: str):
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=filename)
        download_stream = blob_client.download_blob()
        file_data = download_stream.readall()
        return {"filename": filename, "content": file_data.decode('utf-8')}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"File not found: {e}")

@app.post("/process/{filename}")
async def process_file(filename: str):
    try:
        # Fetch the file from Azure Blob
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=filename)
        download_stream = blob_client.download_blob()
        file_data = download_stream.readall()

        # Example processing: Reverse file content
        processed_data = file_data.decode('utf-8')[::-1]

        # Save processed data locally
        processed_filename = f"processed_{filename}"
        with open(processed_filename, "w") as processed_file:
            processed_file.write(processed_data)
        
        return {"message": f"File {filename} processed successfully.", "processed_file": processed_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.post("/upload/{filename}")
async def upload_file(filename: str, processed_file: UploadFile = File(...)):
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=filename)
        blob_client.upload_blob(processed_file.file, overwrite=True)
        return {"message": f"File {filename} uploaded successfully to Azure Blob Storage."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")


# Run the app with uvicorn in your terminal
# uvicorn main:app --reload
