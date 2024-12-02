from client import *
import numpy as np

def test_data_splitting():
    # Load HDF5 files and preprocess data
    EPISODE_PATH = "data/episodes/"  # Update to the correct folder containing .hdf5 files
    preprocessed_data = load_and_preprocess_dynamic_episodes(EPISODE_PATH)

    # Combine all preprocessed data into a dictionary
    data_dict = {
        'controls': preprocessed_data['controls'],
        'frame': preprocessed_data['frames'],
        'hlc': preprocessed_data['hlc'],
        'light': preprocessed_data['light'],
        'measurements': preprocessed_data['measurements'],
        'rgb': preprocessed_data['rgb'],
        'segmentation': preprocessed_data['segmentation'],
    }

    # Split data
    num_models = 3  # Example: 3 local models
    data_distribution = [0.4, 0.4, 0.2]  # Example distribution (model 1 gets 40%, model 2 gets 40%, model 3 gets 20%)
    model_data, val_data, test_data = split_data_for_models_multiple_types(data_dict, num_models, data_distribution)

    # Verify the splits
    print(f"Training data for model 1: {model_data['model_1']['rgb'].shape}")
    print(f"Training data for model 2: {model_data['model_2']['rgb'].shape}")
    print(f"Training data for model 3: {model_data['model_3']['rgb'].shape}")
    print(f"Validation data: {val_data['rgb'].shape}")
    print(f"Test data: {test_data['rgb'].shape}")

if __name__ == "__main__":
    test_data_splitting()
