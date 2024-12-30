import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import re
import numpy as np
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv() 

CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")

if not CLIENT_ACCOUNT_URL:
    logging.error("SAS url environment variable is missing.")
    raise ValueError("Missing required environment variable: SAS url")

try:
    BLOB_SERVICE_CLIENT = BlobServiceClient(account_url=CLIENT_ACCOUNT_URL)
except Exception as e:
    logging.error(f"Failed to initialize Azure Blob Service: {e}")
    raise

# Set up logger
def setup_logger():
    """Set up logging for client."""
    script_directory = os.path.dirname(os.path.realpath(__file__))
    log_file = os.path.join(script_directory, "logs","training.log")
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
def train_model(model, data, epochs=1, batch_size=32):
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

def get_versioned_filename(client_id, save_dir, prefix="local_weights", extension="keras"):
    """
    Generate a versioned filename with timestamp for saving models or weights.

    Args:
        client_id (str): ID of the client.
        save_dir (str): Directory to save files.
        prefix (str): Prefix for the file (e.g., 'weights', 'model').
        extension (str): File extension (e.g., 'keras').

    Returns:
        str: Full path to the versioned filename.
        int: Next version number.
        str: Timestamp for the file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_pattern = re.compile(rf"{prefix}_client{client_id}_v(\d+).*\.{extension}")
    existing_versions = [
        int(version_pattern.match(f).group(1))
        for f in os.listdir(save_dir)
        if version_pattern.match(f)
    ]
    next_version = max(existing_versions, default=0) + 1
    filename = f"{prefix}_client{client_id}_v{next_version}_{timestamp}.{extension}"
    return os.path.join(save_dir, filename), next_version, timestamp


def save_weights(client_id, model, save_dir):
    """
    Save model weights with versioning.
    """
    os.makedirs(save_dir, exist_ok=True)

    weights_path, next_version, timestamp = get_versioned_filename(client_id, save_dir)
    try:
        model.save_weights(weights_path)
        logging.info(f"Weights for {client_id} saved at {weights_path}")
    except Exception as e:
        logging.error(f"Failed to save weights for {client_id}: {e}")
    return weights_path, timestamp


def save_model(client_id, model, save_dir):
    """
    Save the trained model with versioning.
    """
    os.makedirs(save_dir, exist_ok=True)

    model_path, next_version, timestamp = get_versioned_filename(client_id, save_dir)
    try:
        model.save(model_path)
        logging.info(f"Model for {client_id} saved at {model_path}")
    except Exception as e:
        logging.error(f"Failed to save model for {client_id}: {e}")
    return model_path


def upload_file(file_path, container_name):
    """
    Upload a file to Azure Blob Storage with versioned naming.

    Args:
        client_id (str): ID of the client.
        file_path (str): Path to the file to upload.
        container_name (str): Azure container name.
    """
    filename = os.path.basename(file_path)
    try:
        blob_client = BLOB_SERVICE_CLIENT.get_blob_client(container=container_name, blob=filename)
        with open(file_path, "rb") as file:
            blob_client.upload_blob(file.read(), overwrite=True)
        logging.info(f"File {filename} uploaded successfully to Azure Blob Storage.")
        print(f"File {filename} uploaded successfully to Azure Blob Storage.")
    except Exception as e:
        logging.error(f"Error uploading file {filename}: {e}")
        print(f"Error uploading file {filename}: {e}")


# Updated Main Function to Reflect Unified Versioning
def main(client_id, data_path, save_dir, build_model):
    setup_logger()
    logging.info(f"Starting training...")

    try:
        # Load preprocessed data
        data = load_preprocessed_data(data_path)
        logging.info("Data loaded successfully.")
        print("Data loaded successfully")

        # Define the input shapes
        input_shapes = {
            "rgb": (128, 128, 3),
            "segmentation": (128, 128, 3),
            "hlc": (1,),
            "light": (1,),
            "measurements": (1,)
        }

        # Build and train the model
        model = build_model(input_shapes)
        logging.info("Model created successfully.")
        print("model created successfully")

        train_model(model, data)

        # Save and upload weights
        weights_path, timestamp = save_weights(client_id, model, save_dir)
        upload_file(weights_path, CLIENT_CONTAINER_NAME)

        # Save and upload the full model
        # model_path = save_model(client_id, model, save_dir)

        logging.info(f"Training completed successfully.")
        print("Training and upload completed successfully.")
    
    except Exception as e:
        logging.error(f"Error during training: {e}")
        print(f"Error during training: {e}")