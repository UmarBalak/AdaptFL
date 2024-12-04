import numpy as np
import os
import h5py

def split_data_for_clients(client_data_path, val_split=0.2, test_split=0.1, random_seed=None):
    """
    Splits the data of a client into training, validation, and test sets.

    Parameters:
    - client_data_path: Path to the client's .hdf5 data file.
    - val_split: Proportion of data to be used for validation.
    - test_split: Proportion of data to be used for testing.
    - random_seed: Seed for reproducibility.

    Returns:
    - train_data, val_data, test_data: Data splits as dictionaries.
    """

    # Load the data from the client's .hdf5 file
    with h5py.File(client_data_path, 'r') as f:
        # Extract data arrays
        controls = f['controls'][:]
        frames = f['frame'][:]
        hlc = f['hlc'][:]
        light = f['light'][:]
        measurements = f['measurements'][:]
        rgb = f['rgb'][:]
        segmentation = f['segmentation'][:]

    # Get total samples count
    total_samples = len(rgb)

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Shuffle indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    # Calculate split sizes
    val_size = int(total_samples * val_split)
    test_size = int(total_samples * test_split)

    # Split the data indices
    val_indices = indices[:val_size]
    test_indices = indices[val_size:val_size + test_size]
    train_indices = indices[val_size + test_size:]

    # Split data into training, validation, and test sets
    train_data = {
        'controls': controls[train_indices],
        'frames': frames[train_indices],
        'hlc': hlc[train_indices],
        'light': light[train_indices],
        'measurements': measurements[train_indices],
        'rgb': rgb[train_indices],
        'segmentation': segmentation[train_indices]
    }

    val_data = {
        'controls': controls[val_indices],
        'frames': frames[val_indices],
        'hlc': hlc[val_indices],
        'light': light[val_indices],
        'measurements': measurements[val_indices],
        'rgb': rgb[val_indices],
        'segmentation': segmentation[val_indices]
    }

    test_data = {
        'controls': controls[test_indices],
        'frames': frames[test_indices],
        'hlc': hlc[test_indices],
        'light': light[test_indices],
        'measurements': measurements[test_indices],
        'rgb': rgb[test_indices],
        'segmentation': segmentation[test_indices]
    }

    return train_data, val_data, test_data

# Example Usage:
client_data_path = 'data/client_1/data.hdf5'
train_data, val_data, test_data = split_data_for_clients(client_data_path)

client_data_path = 'data/client_1/data.hdf5'
train_data, val_data, test_data = split_data_for_clients(client_data_path)

client_data_path = 'data/client_1/data.hdf5'
train_data, val_data, test_data = split_data_for_clients(client_data_path)
