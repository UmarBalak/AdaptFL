import logging
import numpy as np
from .global_server import validate_model_weights, evaluate_global_model, distribute_global_model
from local_training import train_local_models
from aggregate_weights import aggregate_weights

def main():
    # Paths and directories
    global_model_path = "../AdaptFL_Project/global_server/global_model/global_model.h5"
    client_dirs = ["../AdaptFL_Project/client1", "../AdaptFL_Project/client2", "../AdaptFL_Project/client3"]
    test_data_path = "../AdaptFL_Project/test_data/"

    # Step 1: Distribute the global model
    logging.info("Step 1: Distributing global model to all clients.")
    distribute_global_model(global_model_path, client_dirs)

    # Step 2: Validate model consistency
    logging.info("Step 2: Validating model consistency across clients.")
    validate_model_weights(global_model_path, client_dirs)

    # Step 3: Local training
    logging.info("Step 3: Starting local training for each client.")
    train_local_models(client_dirs)

    # Step 4: Aggregate client weights
    logging.info("Step 4: Aggregating weights from all clients.")
    aggregate_weights(client_dirs, global_model_path)

    # Step 5: Evaluate the aggregated global model
    logging.info("Step 5: Evaluating the aggregated global model.")
    test_data = {
        "rgb": np.load(test_data_path + "rgb.npy"),
        "segmentation": np.load(test_data_path + "segmentation.npy"),
        "controls": np.load(test_data_path + "controls.npy"),
    }
    evaluate_global_model(global_model_path, test_data)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
