import os
import logging
from keras.models import load_model
import numpy as np

# Setup logger
def setup_logger():
    logging.basicConfig(
        filename="global_server/logs/debug_weights.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def log_weight_shapes(model_path, label):
    """
    Logs the shape of each layer's weights in a model.

    Args:
        model_path (str): Path to the model.
        label (str): Label to identify the model in logs.
    """
    try:
        model = load_model(model_path)
        logging.info(f"{label} - Model loaded successfully.")
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                shapes = [w.shape for w in weights]
                logging.info(f"{label} - Layer {layer.name} weights: {shapes}")
            else:
                logging.info(f"{label} - Layer {layer.name} has no weights.")
    except Exception as e:
        logging.error(f"{label} - Error loading model: {e}")

if __name__ == "__main__":
    setup_logger()

    # Paths
    global_model_path = "../AdaptFL_Project/global_server/global_model/global_model.h5"
    client_model_paths = [
        "../AdaptFL_Project/client1/models/global_model.h5",
        "../AdaptFL_Project/client2/models/global_model.h5",
        "../AdaptFL_Project/client3/models/global_model.h5",
    ]

    # Log global model weights
    log_weight_shapes(global_model_path, "Global Model")

    # Log client model weights
    for i, client_model_path in enumerate(client_model_paths):
        log_weight_shapes(client_model_path, f"Client Model {i+1}")
