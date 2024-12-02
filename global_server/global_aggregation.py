import tensorflow as tf
import numpy as np
import os

def load_model_weights(model_path):
    """
    Load the weights from a saved model.

    Parameters:
    - model_path: Path to the saved model file.

    Returns:
    - model_weights: List of model weights.
    """
    model = tf.keras.models.load_model(model_path)
    return model.get_weights()


def aggregate_weights(local_weights, data_proportions):
    """
    Aggregate local model weights using the FedAvg algorithm.

    Parameters:
    - local_weights: List of weight lists from local models.
    - data_proportions: List of data proportions corresponding to each model.

    Returns:
    - aggregated_weights: List of aggregated weights.
    """
    aggregated_weights = [
        np.zeros_like(weight) for weight in local_weights[0]
    ]

    for model_weights, proportion in zip(local_weights, data_proportions):
        for i, layer_weights in enumerate(model_weights):
            aggregated_weights[i] += proportion * layer_weights

    return aggregated_weights


def update_global_model(global_model, aggregated_weights):
    """
    Update the global model with aggregated weights.

    Parameters:
    - global_model: The global model instance.
    - aggregated_weights: List of aggregated weights.

    Returns:
    - global_model: Updated global model.
    """
    global_model.set_weights(aggregated_weights)
    return global_model


def aggregate_and_save_global_model(local_model_paths, data_proportions, save_path="global_model.h5"):
    """
    Aggregate local model weights, update the global model, and save it.

    Parameters:
    - local_model_paths: List of paths to the saved local models.
    - data_proportions: List of data proportions for each local model.
    - save_path: Path to save the global model.

    Returns:
    - global_model: The updated global model.
    """
    # Load weights from local models
    local_weights = [load_model_weights(path) for path in local_model_paths]

    # Aggregate weights
    aggregated_weights = aggregate_weights(local_weights, data_proportions)

    # Build a global model structure (matching local model structure)
    global_model = tf.keras.models.load_model(local_model_paths[0])  # Use structure of the first model

    # Update global model with aggregated weights
    global_model = update_global_model(global_model, aggregated_weights)

    # Save global model
    global_model.save(save_path)
    print(f"Global model saved at {save_path}")

    return global_model
