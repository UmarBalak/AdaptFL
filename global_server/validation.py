import os
import logging
from keras.models import load_model
import numpy as np

# Setup logger
def setup_logger():
    logging.basicConfig(
        filename="validation.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def validate_model_existence(client_dirs, global_model_path):
    """
    Validate that the global model exists in all client directories.

    Args:
        client_dirs (list): List of client directories.
        global_model_path (str): Path to the global model.
    """
    setup_logger()
    for client_dir in client_dirs:
        client_model_path = os.path.join(client_dir, "models", "global_model.h5")
        if not os.path.exists(client_model_path):
            logging.error(f"Global model missing at {client_model_path}")
        else:
            logging.info(f"Global model exists at {client_model_path}")

def validate_model_integrity(global_model_path, client_dirs):
    """
    Validate that the model architecture and weights are intact.

    Args:
        global_model_path (str): Path to the global model on the server.
        client_dirs (list): List of client directories to validate.
    """
    try:
        # Load the global model to check architecture and weights
        global_model = load_model(global_model_path)
        logging.info("Global model architecture and weights are valid.")

        # Validate model architecture consistency at clients
        for client_dir in client_dirs:
            client_model_path = os.path.join(client_dir, "models", "global_model.h5")
            if os.path.exists(client_model_path):
                try:
                    client_model = load_model(client_model_path)
                    if global_model.to_json() == client_model.to_json():
                        logging.info(f"Model architecture matches at {client_model_path}")
                    else:
                        logging.error(f"Model architecture mismatch at {client_model_path}")
                except Exception as e:
                    logging.error(f"Error loading model at {client_model_path}: {e}")
            else:
                logging.error(f"Global model missing at {client_model_path}")
    except Exception as e:
        logging.error(f"Error loading global model: {e}")

def validate_model_weights(global_model_path, client_dirs):
    """
    Validate that model weights are consistent across clients.

    Args:
        global_model_path (str): Path to the global model on the server.
        client_dirs (list): List of client directories to validate.
    """
    try:
        global_model = load_model(global_model_path)
        global_weights = global_model.get_weights()
        logging.info("Global model weights shapes:")
        for idx, weight in enumerate(global_weights):
            logging.info(f"Global Weight[{idx}]: {weight.shape}")

        for client_dir in client_dirs:
            client_model_path = os.path.join(client_dir, "models", "global_model.h5")
            if os.path.exists(client_model_path):
                try:
                    client_model = load_model(client_model_path)
                    client_weights = client_model.get_weights()

                    logging.info(f"Client weights shapes at {client_model_path}:")
                    for idx, weight in enumerate(client_weights):
                        logging.info(f"Client Weight[{idx}]: {weight.shape}")

                    # Compare weights shape
                    for idx, (gw, cw) in enumerate(zip(global_weights, client_weights)):
                        if gw.shape != cw.shape:
                            logging.error(
                                f"Weight shape mismatch at layer {idx}: "
                                f"Global: {gw.shape}, Client: {cw.shape} at {client_model_path}"
                            )
                        elif not np.allclose(gw, cw):
                            logging.warning(
                                f"Weight values mismatch at layer {idx}: "
                                f"Client: {client_model_path}"
                            )
                        else:
                            logging.info(f"Layer {idx} weights match.")
                except Exception as e:
                    logging.error(f"Error loading model weights at {client_model_path}: {e}")
            else:
                logging.error(f"Global model missing at {client_model_path}")
    except Exception as e:
        logging.error(f"Error loading global model: {e}")

if __name__ == "__main__":
    # Paths
    client_dirs = [
        "../AdaptFL_Project/client1",
        "../AdaptFL_Project/client2",
        "../AdaptFL_Project/client3",
    ]
    global_model_path = "../AdaptFL_Project/global_server/global_model/global_model.h5"

    # Validate model existence, architecture, and weights
    validate_model_existence(client_dirs, global_model_path)
    validate_model_integrity(global_model_path, client_dirs)
    validate_model_weights(global_model_path, client_dirs)
