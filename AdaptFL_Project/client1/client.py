import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import numpy as np
from keras import layers, models


def setup_logger(client_id):
    """Set up logging for client."""
    log_file = f"../AdaptFL_Project/{client_id}/logs/training.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def load_preprocessed_data(client_id):
    """Load preprocessed data for training."""
    preprocessed_data_path = f"../AdaptFL_Project/{client_id}/preprocessed_data"
    data = {}

    for dataset_name in ["rgb", "segmentation", "controls", "frames", "hlc", "light", "measurements"]:
        data[dataset_name] = np.load(os.path.join(preprocessed_data_path, f"{dataset_name}.npy"))

    return data

from keras import layers, models

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

def train_model(model, data, epochs=5, batch_size=32):
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
        verbose=1  # Enable verbose output for training process
    )


def save_model(client_id, model):
    """Save the trained model."""
    model_path = f"../AdaptFL_Project/{client_id}/models/trained_model.h5"
    model.save(model_path)
    logging.info(f"Model saved at {model_path}")

def save_model_weights(client_id, model):
    """Save the model weights locally for future upload to global server."""
    model_weights_path = f"../AdaptFL_Project/{client_id}/models/model_weights.h5"
    model.save_weights(model_weights_path)
    logging.info(f"Model weights saved at {model_weights_path}")

def main(client_id):
    """Main function to execute client training pipeline."""
    # setup_logger(client_id)
    logging.info(f"Starting training for {client_id}")

    try:
        # Load preprocessed data
        data = load_preprocessed_data(client_id)
        print("Data loaded successfully.")

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
        print("Model created successfully")

        train_model(model, data)

        # Save the trained model
        save_model(client_id, model)

        # Save model weights locally (instead of sharing with server yet)
        save_model_weights(client_id, model)

        logging.info(f"Training completed successfully for {client_id}")
    
    except Exception as e:
        logging.error(f"Error during training for {client_id}: {e}")

if __name__ == "__main__":
    print("starting...")
    main("client1")
