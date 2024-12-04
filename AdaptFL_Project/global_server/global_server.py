import os
import shutil
import logging
import numpy as np
from keras import models

def setup_logger():
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        filename="global_model_distribution.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def distribute_global_model(global_model_path, client_dirs):
    """
    Distribute the global model to all clients.

    Args:
        global_model_path (str): Path to the global model.
        client_dirs (list): List of client directories.
    """
    for client_dir in client_dirs:
        client_model_path = os.path.join(client_dir, "models", "global_model.h5")
        try:
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(client_model_path), exist_ok=True)
            shutil.copy(global_model_path, client_model_path)
            logging.info(f"Distributed global model to {client_model_path}")
        except Exception as e:
            logging.error(f"Error distributing model to {client_dir}: {e}")

def evaluate_global_model(global_model_path, test_data):
    """
    Evaluate the global model on test data.

    Args:
        global_model_path (str): Path to the global model.
        test_data (dict): Test data containing inputs and true outputs.

    Returns:
        dict: Evaluation metrics including loss and predictions.
    """
    try:
        # Load the global model
        model = models.load_model(global_model_path)
        logging.info(f"Loaded global model from {global_model_path}")

        # Check if all required inputs are available
        required_inputs = ["rgb", "segmentation", "hlc", "light", "measurements", "controls"]
        missing_inputs = [key for key in required_inputs if key not in test_data]
        if missing_inputs:
            logging.error(f"Missing test data inputs: {missing_inputs}")
            return None

        # Prepare inputs and target
        test_inputs = [
            test_data["rgb"],
            test_data["segmentation"],
            test_data["hlc"],
            test_data["light"],
            test_data["measurements"]
        ]
        test_targets = test_data["controls"]

        # Evaluate the model
        loss = model.evaluate(test_inputs, test_targets, verbose=1)
        predictions = model.predict(test_inputs)

        # Log and return metrics
        logging.info(f"Model evaluation completed. Loss: {loss}")
        logging.info(f"Prediction shape: {predictions.shape}")

        return {
            "loss": loss,
            "predictions": predictions
        }
    except Exception as e:
        logging.error(f"Failed to evaluate the global model: {e}")
        return None

def load_test_data(test_data_path):
    """
    Load test data from specified directory.

    Args:
        test_data_path (str): Path to the directory containing test data.

    Returns:
        dict: Test data loaded from files, or None if errors occur.
    """
    test_data = {}
    for dataset in ["rgb", "segmentation", "hlc", "light", "measurements", "controls"]:
        data_file = os.path.join(test_data_path, f"{dataset}.npy")
        if os.path.exists(data_file):
            test_data[dataset] = np.load(data_file)
        else:
            logging.error(f"Missing test data file: {data_file}")
            return None
    return test_data

def main():
    # Setup logging
    setup_logger()

    # Paths and directories
    client_dirs = [
        "../AdaptFL_Project/client1",
        "../AdaptFL_Project/client2",
        "../AdaptFL_Project/client3"
    ]
    global_model_path = "../AdaptFL_Project/global_server/global_model/global_model.h5"
    test_data_path = "../AdaptFL_Project/test_data/processed_test_data"

    # Check if the global model exists
    if not os.path.exists(global_model_path):
        logging.error(f"Global model file not found: {global_model_path}")
        return

    # Distribute the global model to clients
    distribute_global_model(global_model_path, client_dirs)

    # Load test data
    test_data = load_test_data(test_data_path)
    if test_data is None:
        logging.error("Test data loading failed.")
        return

    # Evaluate the global model on the test data
    evaluation_metrics = evaluate_global_model(global_model_path, test_data)
    if evaluation_metrics:
        logging.info(f"Evaluation Results: Loss - {evaluation_metrics['loss']}")
        logging.info("Model evaluation completed successfully.")

if __name__ == "__main__":
    main()
