import os
import logging
import numpy as np
from keras import models

def setup_logger():
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        filename="logs/client_model_validation.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def load_test_data(test_data_path):
    """
    Load test data from the specified directory.

    Args:
        test_data_path (str): Path to the directory containing test data.

    Returns:
        dict: Test data containing required inputs and target outputs.
    """
    test_data = {}
    required_files = ["rgb", "segmentation", "hlc", "light", "measurements", "controls"]

    for dataset in required_files:
        file_path = os.path.join(test_data_path, f"{dataset}.npy")
        if os.path.exists(file_path):
            test_data[dataset] = np.load(file_path)
        else:
            logging.error(f"Test data file missing: {file_path}")
            raise FileNotFoundError(f"Missing required test data file: {file_path}")

    return test_data

def validate_client_model(client_id, model_path, test_data):
    """
    Validate a single client model against the test dataset.

    Args:
        client_id (str): ID of the client whose model is being validated.
        model_path (str): Path to the client's model.
        test_data (dict): Preloaded test data.

    Returns:
        dict: Evaluation metrics including loss and predictions.
    """
    try:
        # Load client model
        model = models.load_model(model_path)
        logging.info(f"Loaded model for {client_id} from {model_path}")

        # Validate test data integrity
        required_inputs = ["rgb", "segmentation", "hlc", "light", "measurements", "controls"]
        for key in required_inputs:
            if key not in test_data:
                raise ValueError(f"Missing required test data key: {key}")

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
        loss = model.evaluate(test_inputs, test_targets, verbose=0)
        predictions = model.predict(test_inputs)

        # Log and return metrics
        logging.info(f"{client_id} - Loss: {loss}")
        logging.info(f"{client_id} - Predictions shape: {predictions.shape}")

        return {
            "client_id": client_id,
            "loss": loss,
            "predictions": predictions
        }
    except Exception as e:
        logging.error(f"Validation failed for {client_id}: {e}")
        return {
            "client_id": client_id,
            "error": str(e)
        }

def main():
    """
    Main function to validate all client models against a standardized test dataset.
    """
    # Setup logging
    setup_logger()

    # Paths and directories
    client_dirs = [
        "../AdaptFL_Project/client1",
        "../AdaptFL_Project/client2",
        "../AdaptFL_Project/client3"
    ]
    test_data_path = "../AdaptFL_Project/test_data/processed_test_data"
    validation_results = []

    # Load test data
    try:
        test_data = load_test_data(test_data_path)
        logging.info("Test data loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Test data loading failed: {e}")
        return

    # Validate each client's model
    for client_dir in client_dirs:
        client_id = os.path.basename(client_dir)
        model_path = os.path.join(client_dir, "models", "trained_model.h5")

        if os.path.exists(model_path):
            result = validate_client_model(client_id, model_path, test_data)
            validation_results.append(result)
        else:
            logging.error(f"Model file not found for {client_id}: {model_path}")
            validation_results.append({
                "client_id": client_id,
                "error": f"Model file missing: {model_path}"
            })

    # Summarize results
    logging.info("Validation Summary:")
    for result in validation_results:
        if "error" in result:
            logging.error(f"{result['client_id']} - Error: {result['error']}")
        else:
            logging.info(f"{result['client_id']} - Loss: {result['loss']}")

if __name__ == "__main__":
    main()
