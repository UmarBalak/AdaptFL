import os
import shutil
import logging
import numpy as np
import gzip
from datetime import datetime
from keras import models

def setup_logger():
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        filename="global_server/logs/global_model_distribution.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def compress_model(model_path, compressed_model_path):
    """
    Compress the global model file to reduce size for distribution.

    Args:
        model_path (str): Path to the model file.
        compressed_model_path (str): Path to save the compressed model.
    """
    try:
        with open(model_path, "rb") as f_in, gzip.open(compressed_model_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        logging.info(f"Compressed model saved at {compressed_model_path}")
    except Exception as e:
        logging.error(f"Error compressing model: {e}")

def distribute_global_model(global_model_path, client_dirs):
    """
    Distribute the global model to all clients.

    Args:
        global_model_path (str): Path to the global model.
        client_dirs (list): List of client directories.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        compressed_model_path = f"{global_model_path}_{timestamp}.gz"
        compress_model(global_model_path, compressed_model_path)

        for client_dir in client_dirs:
            try:
                client_model_path = os.path.join(client_dir, "models", "global", f"global_model_{timestamp}.h5.gz")
                os.makedirs(os.path.dirname(client_model_path), exist_ok=True)
                shutil.copy(compressed_model_path, client_model_path)
                logging.info(f"Distributed global model to {client_model_path}")
            except Exception as e:
                logging.error(f"Error distributing model to {client_dir}: {e}")
    except Exception as e:
        logging.error(f"Global model distribution failed: {e}")

def evaluate_global_model(global_model_path, test_data, save_predictions=False):
    """
    Evaluate the global model on test data.

    Args:
        global_model_path (str): Path to the global model.
        test_data (dict): Test data containing inputs and true outputs.
        save_predictions (bool): Whether to save predictions to a file.

    Returns:
        dict: Evaluation metrics including loss and predictions.
    """
    try:
        model = models.load_model(global_model_path)
        logging.info(f"Loaded global model from {global_model_path}")

        test_inputs = [
            test_data["rgb"],
            test_data["segmentation"],
            test_data["hlc"],
            test_data["light"],
            test_data["measurements"]
        ]
        test_targets = test_data["controls"]

        loss = model.evaluate(test_inputs, test_targets, verbose=1)
        predictions = model.predict(test_inputs)

        if save_predictions:
            predictions_path = os.path.join(os.path.dirname(global_model_path), "predictions.npy")
            np.save(predictions_path, predictions)
            logging.info(f"Predictions saved at {predictions_path}")

        return {"loss": loss, "predictions": predictions}
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
    try:
        test_data = {}
        for dataset in ["rgb", "segmentation", "hlc", "light", "measurements", "controls"]:
            data_file = os.path.join(test_data_path, f"{dataset}.npy")
            if os.path.exists(data_file):
                test_data[dataset] = np.load(data_file)
            else:
                logging.error(f"Missing test data file: {data_file}")
                return None
        return test_data
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        return None

def main():
    # Setup logging
    setup_logger()

    # Paths and directories
    client_dirs = [
        "../AdaptFL_Project/client1",
        "../AdaptFL_Project/client2",
        "../AdaptFL_Project/client3"
    ]
    global_model_dir = "../AdaptFL_Project/global_server/global_model/"
    test_data_path = "../AdaptFL_Project/test_data/processed_test_data"

    # Validate global model directory existence
    if not os.path.exists(global_model_dir) or not os.path.isdir(global_model_dir):
        logging.error(f"Global model directory not found or is not a directory: {global_model_dir}")
        return

    # Identify the most recent global model file
    try:
        global_model_files = [f for f in os.listdir(global_model_dir) if f.endswith(".h5")]
        if not global_model_files:
            logging.error(f"No global model file found in directory: {global_model_dir}")
            return
        
        global_model_files.sort()
        global_model_file = global_model_files[-1]  # Latest model based on name
        global_model_path = os.path.join(global_model_dir, global_model_file)
    except Exception as e:
        logging.error(f"Error identifying the global model file: {e}")
        return

    # Distribute the model to clients
    try:
        distribute_global_model(global_model_path, client_dirs)
        logging.info(f"Distributed global model: {global_model_path}")
    except Exception as e:
        logging.error(f"Error distributing global model: {e}")
        return

    # Load and evaluate the global model
    test_data = load_test_data(test_data_path)
    if test_data:
        evaluation_metrics = evaluate_global_model(global_model_path, test_data, save_predictions=True)
        if evaluation_metrics:
            logging.info(f"Evaluation Results - Loss: {evaluation_metrics['loss']}")
        else:
            logging.error("Model evaluation failed.")
    else:
        logging.error("Test data loading failed.")

    # Additional logging for debugging purposes
    logging.debug(f"Client directories: {client_dirs}")
    logging.debug(f"Global model directory: {global_model_dir}")
    logging.debug(f"Test data path: {test_data_path}")

if __name__ == "__main__":
    main()
