import os
import numpy as np
from models import build_model

def train_local_model(model_id, train_data, val_data, epochs=10, batch_size=32, save_dir="saved_models"):
    """
    Trains a local model on its assigned data.

    Parameters:
    - model_id: Identifier for the local model (e.g., "model_1").
    - train_data: Dictionary containing training inputs and labels.
    - val_data: Dictionary containing validation inputs and labels.
    - epochs: Number of training epochs.
    - batch_size: Batch size for training.
    - save_dir: Directory to save the trained model.

    Returns:
    - history: Training history object.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Extract training and validation data
    train_inputs = {
        "image_input": train_data["rgb"],
        "aux_input": np.hstack((
            train_data["frames"], 
            train_data["hlc"], 
            train_data["light"], 
            train_data["measurements"]
        ))
    }
    train_labels = train_data["controls"]

    val_inputs = {
        "image_input": val_data["rgb"],
        "aux_input": np.hstack((
            val_data["frames"], 
            val_data["hlc"], 
            val_data["light"], 
            val_data["measurements"]
        ))
    }
    val_labels = val_data["controls"]

    # Build model
    model = build_model()

    # Train the model
    history = model.fit(
        train_inputs, train_labels,
        validation_data=(val_inputs, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )

    # Save the model
    model_path = os.path.join(save_dir, f"{model_id}.h5")
    model.save(model_path)
    print(f"Model {model_id} saved at {model_path}")

    return history
