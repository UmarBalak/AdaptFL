import os
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime
from model_utils import build_model

# Setup logging
logging.basicConfig(
    filename="global_server/logs/fedavg.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Example usage with input shapes
input_shapes = {
    "rgb": (128, 128, 3),  # Adjust size as per preprocessing
    "segmentation": (128, 128, 3),
    "hlc": (1,),
    "light": (1,),
    "measurements": (1,)
}
model = build_model(input_shapes)

def load_model_weights(client_dirs):
    """
    Dynamically load model weights from all clients.

    Args:
        client_dirs (list): List of client directories.

    Returns:
        weights_list (list): List of model weights as numpy arrays.
    """
    weights_list = []
    for client_dir in client_dirs:
        try:
            # Dynamically find the latest saved model file
            model_dir = os.path.join(client_dir, "models/local")
            if not os.path.exists(model_dir):
                logging.warning(f"Model directory not found: {model_dir}")
                continue
            
            # Find the most recent model file based on timestamp in filename
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
            if not model_files:
                logging.warning(f"No model files found in {model_dir}")
                continue

            latest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
            model_path = os.path.join(model_dir, latest_model)
            
            # Load the model and append its weights
            client_model = tf.keras.models.load_model(model_path)
            weights_list.append(client_model.get_weights())
            logging.info(f"Successfully loaded model weights from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model weights for {client_dir}: {e}")
    return weights_list

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

def update_global_model(global_model_dir, avg_weights, input_shapes):
    """
    Save the global model with aggregated weights and a timestamp for versioning.

    Args:
        global_model_dir (str): Directory to save the global model.
        avg_weights (list): Federated averaged weights.
        input_shapes (dict): Input shapes for building the global model.
    """
    os.makedirs(global_model_dir, exist_ok=True)  # Ensure directory exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_model_path = os.path.join(global_model_dir, f"global_model_v{timestamp}.h5")

    # Build, update, and save the global model
    model = build_model(input_shapes)
    model.set_weights(avg_weights)
    model.save(global_model_path)
    logging.info(f"Updated global model saved at {global_model_path}")

def main():
    """
    Main function to execute global server tasks.
    """
    try:
        logging.info("Starting global server aggregation process.")

        # Define client directories
        client_dirs = [
            "../AdaptFL_Project/client1",
            "../AdaptFL_Project/client2",
            "../AdaptFL_Project/client3",
        ]

        # Load client weights
        weights_list = load_model_weights(client_dirs)

        # Perform Federated Averaging
        if weights_list:
            avg_weights = federated_averaging(weights_list)

            # Update global model
            global_model_path = "../AdaptFL_Project/global_server/global_model"
            input_shapes = {
                "rgb": (128, 128, 3),  
                "segmentation": (128, 128, 3),
                "hlc": (1,),
                "light": (1,),
                "measurements": (1,)
            }
            update_global_model(global_model_path, avg_weights, input_shapes)

            logging.info("Global model aggregation completed successfully.")
        else:
            logging.error("No valid weights to aggregate.")

    except Exception as e:
        logging.error(f"Error in global server: {e}")

if __name__ == "__main__":
    main()
