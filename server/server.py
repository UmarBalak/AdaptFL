import os
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from azure.storage.blob import BlobServiceClient
from keras.models import load_model

# Initialize FastAPI app
app = FastAPI()

# Azure Blob Storage configuration
CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")
SERVER_CONTAINER_NAME = os.getenv("SERVER_CONTAINER_NAME")
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

def federated_averaging(weights_list):
    """
    Perform Federated Averaging on a list of model weights.

    Args:
        weights_list (list): List of model weights from clients.

    Returns:
        avg_weights (list): Federated averaged weights.
    """
    if not weights_list:
        logging.error("No weights available for aggregation")
        return None

    avg_weights = []
    for layer_weights in zip(*weights_list):  # Iterate layer-wise
        avg_weights.append(np.mean(layer_weights, axis=0))  # Average weights for each layer
    logging.info("Federated averaging completed successfully.")
    return avg_weights

def load_weights_from_blob(blob_client):
    """
    Load model weights from a blob.

    Args:
        blob_client: Blob client to download the weights.

    Returns:
        weights (list): Model weights.
    """
    with open("temp_model.keras", "wb") as file:
        download_stream = blob_client.download_blob()
        file.write(download_stream.readall())
    
    model = load_model("temp_model.keras")
    weights = model.get_weights()
    os.remove("temp_model.keras")
    return weights

def save_weights_to_blob(weights, filename):
    """
    Save model weights to a blob.

    Args:
        weights (list): Model weights to save.
        filename (str): Name of the file to save the weights as.
    """
    model = load_model("temp_model.keras")
    model.set_weights(weights)
    model.save("temp_model.keras")

    blob_client = blob_service_client.get_blob_client(container=SERVER_CONTAINER_NAME, blob=filename)
    with open("temp_model.keras", "rb") as file:
        blob_client.upload_blob(file, overwrite=True)
    
    os.remove("temp_model.keras")

@app.get("/aggregate-weights")
async def aggregate_weights():
    try:
        container_client = blob_service_client.get_container_client(CLIENT_CONTAINER_NAME)
        blob_list = container_client.list_blobs(name_starts_with="client")

        weights_list = []
        for blob in blob_list:
            if blob.name.endswith(".keras"):
                blob_client = container_client.get_blob_client(blob)
                weights = load_weights_from_blob(blob_client)
                weights_list.append(weights)

        if not weights_list:
            raise HTTPException(status_code=404, detail="No .keras files found in the container")

        avg_weights = federated_averaging(weights_list)
        if avg_weights is None:
            raise HTTPException(status_code=500, detail="Federated averaging failed")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aggregated_weights_{timestamp}.keras"
        save_weights_to_blob(avg_weights, filename)

        return {"message": "Federated averaging completed and weights saved successfully", "filename": filename}
    
    except Exception as e:
        logging.error(f"Error during aggregation: {e}")
        raise HTTPException(status_code=500, detail="Error during aggregation")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)