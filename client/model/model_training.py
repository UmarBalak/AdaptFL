import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import numpy as np
from keras import layers, models
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv() 

CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")
print(CONNECTION_STRING, CLIENT_CONTAINER_NAME)
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

def upload_file(filename: str, file_content):
    try:
        blob_client = blob_service_client.get_blob_client(container=CLIENT_CONTAINER_NAME, blob=filename)
        blob_client.upload_blob(file_content, overwrite=True)
        logging.info(f"File {filename} uploaded successfully to Azure Blob Storage.")
        print(f"File {filename} uploaded successfully to Azure Blob Storage.")
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        print(f"Error uploading file: {e}")

# Set up logger
def setup_logger(client_id, log_dir):
    """Set up logging for client."""
    log_file = os.path.join(log_dir, f"{client_id}_training.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

# Load preprocessed data
def load_preprocessed_data(data_path):
    """Load preprocessed data for training from a .npz file."""
    data_file = os.path.join(data_path, "preprocessed_data.npz")

    # Load the .npz file
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found.")
    
    data = np.load(data_file)

    # Convert npz data to a dictionary
    loaded_data = {key: data[key] for key in data.keys()}

    return loaded_data

# Train the model
def train_model(model, data, epochs=3, batch_size=32):
    """Train the model on the preprocessed data."""
    rgb_data = data["rgb"]
    segmentation_data = data["segmentation"]
    hlc_data = data["hlc"]
    light_data = data["light"]
    measurements_data = data["measurements"]
    controls_data = data["controls"]  # Target variable

    model.fit(
        [rgb_data, segmentation_data, hlc_data, light_data, measurements_data],  # Inputs
        controls_data,  # Target (throttle, steer, brake)
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

# Save model weights with versioning
def save_weights(client_id, model, save_dir):
    """
    Save the trained model weights with a timestamp for versioning.

    Args:
        client_id (str): The ID of the client (e.g., 'client1').
        model: The trained model whose weights are to be saved.
        save_dir: Directory to save weights.
    """
    # Create a versioned filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_dir = os.path.join(save_dir, "client", client_id, "models/local")
    os.makedirs(weights_dir, exist_ok=True)  # Ensure the directory exists
    weights_path = os.path.join(weights_dir, f"client{client_id}_weights_{timestamp}.keras")
    
    # Save the weights
    try:
        model.save_weights(weights_path)
        logging.info(f"Weights for {client_id} saved at {weights_path}")
    except Exception as e:
        logging.error(f"Failed to save weights for {client_id}: {e}")

    return weights_path, timestamp

# Save the trained model with versioning
def save_model(client_id, model, save_dir):
    """
    Save the trained model with a timestamp for versioning.

    Args:
        client_id (str): The ID of the client (e.g., 'client1').
        model: The trained model to be saved.
        save_dir: Directory to save the model.
    """
    # Create a versioned filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, "client", client_id, "models/local")
    os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
    model_path = os.path.join(model_dir, f"client{client_id}_trained_model_{timestamp}.keras")
    
    # Save the model
    try:
        model.save(model_path)
        logging.info(f"Model for {client_id} saved at {model_path}")
    except Exception as e:
        logging.error(f"Failed to save model for {client_id}: {e}")

    return model_path

# Main function for training
def main(client_id, data_path, save_dir, build_model, epochs=5, batch_size=32):
    """Main function to execute client training pipeline."""
    setup_logger(client_id, save_dir)
    logging.info(f"Starting training for {client_id}")

    try:
        # Load preprocessed data
        data = load_preprocessed_data(data_path)
        logging.info("Data loaded successfully.")

        # Define the input shapes
        input_shapes = {
            "rgb": (128, 128, 3),  # Adjust size as per preprocessing
            "segmentation": (128, 128, 3),
            "hlc": (1,),
            "light": (1,),
            "measurements": (1,)
        }

        # Build and train the model
        model = build_model(input_shapes)
        logging.info("Model created successfully.")

        train_model(model, data, epochs, batch_size)

        # Save the trained model and weights
        model_path = save_model(client_id, model, save_dir)
        weights_path, timestamp = save_weights(client_id, model, save_dir)
        print("Model and weights saved successfully.")
        # Upload the saved weights file
        with open(weights_path, "rb") as file:
            upload_file(f"client{client_id}_local_weights_{timestamp}.keras", file.read())
        logging.info(f"Training completed successfully for {client_id}")
    
    except Exception as e:
        logging.error(f"Error during training for {client_id}: {e}")