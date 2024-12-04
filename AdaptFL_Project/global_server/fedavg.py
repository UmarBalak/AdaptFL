import os
import numpy as np
import tensorflow as tf
from keras import models, layers
import logging

# Setup logging
logging.basicConfig(
    filename="fedavg.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def build_model(input_shapes):
    """
    Builds a multi-input model for predicting controls using multiple modalities.

    Parameters:
    - input_shapes: Dictionary of input shapes for each modality.

    Returns:
    - model: A Keras model.
    """
    inputs = []

    # RGB images input
    rgb_input = layers.Input(shape=input_shapes['rgb'], name="rgb")
    rgb_branch = layers.Conv2D(32, (3, 3), activation='relu')(rgb_input)
    rgb_branch = layers.MaxPooling2D()(rgb_branch)
    rgb_branch = layers.Flatten()(rgb_branch)
    inputs.append(rgb_input)

    # Segmentation masks input
    segmentation_input = layers.Input(shape=input_shapes['segmentation'], name="segmentation")
    segmentation_branch = layers.Conv2D(32, (3, 3), activation='relu')(segmentation_input)
    segmentation_branch = layers.MaxPooling2D()(segmentation_branch)
    segmentation_branch = layers.Flatten()(segmentation_branch)
    inputs.append(segmentation_input)

    # High-Level Command input
    hlc_input = layers.Input(shape=input_shapes['hlc'], name="hlc")
    hlc_branch = layers.Dense(32, activation='relu')(hlc_input)
    inputs.append(hlc_input)

    # Traffic Light Status input
    light_input = layers.Input(shape=input_shapes['light'], name="light")
    light_branch = layers.Dense(32, activation='relu')(light_input)
    inputs.append(light_input)

    # Measurements input (speed)
    measurements_input = layers.Input(shape=input_shapes['measurements'], name="measurements")
    measurements_branch = layers.Dense(32, activation='relu')(measurements_input)
    inputs.append(measurements_input)

    # Concatenate all branches
    concatenated = layers.concatenate([rgb_branch, segmentation_branch, hlc_branch, light_branch, measurements_branch])

    # Output layer (for control predictions)
    output = layers.Dense(3, activation='tanh', name="controls")(concatenated)

    # Define the model
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

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
    Load model weights from all clients and extract weights.

    Args:
        client_dirs (list): List of client directories.

    Returns:
        weights (list): List of model weights as numpy arrays.
    """
    weights_list = []
    for client_dir in client_dirs:
        try:
            weights_path = os.path.join(client_dir, "models", "model_weights.h5")
            if os.path.exists(weights_path):
                # Consider adding a validation check for weight compatibility
                model.load_weights(weights_path)
                weights_list.append(model.get_weights())
        except Exception as e:
            logging.error(f"Error loading weights for {client_dir}: {e}")
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
    return avg_weights

def update_global_model(global_model_path, avg_weights):
    """
    Update the global model with the aggregated weights.

    Args:
        global_model_path (str): Path to save the global model.
        avg_weights (list): Federated averaged weights.
    """
    # Recreate the global model
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
        avg_weights = federated_averaging(weights_list)

        # Update global model
        global_model_path = "../AdaptFL_Project/global_server/global_model/global_model.h5"
        update_global_model(global_model_path, avg_weights)

        logging.info("Global model aggregation completed successfully.")
    
    except Exception as e:
        logging.error(f"Error in global server: {e}")

if __name__ == "__main__":
    main()
