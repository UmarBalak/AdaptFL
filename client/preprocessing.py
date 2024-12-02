import h5py
import numpy as np
import os
import cv2

def load_and_preprocess_dynamic_episodes(directory_path, image_size=(128, 128)):
    """
    Loads all HDF5 files in the specified directory and preprocesses them.

    Parameters:
    - directory_path: Path to the directory containing the .hdf5 files.
    - image_size: Target size for resizing the images.

    Returns:
    - preprocessed_data: A dictionary containing preprocessed data.
    """
    hdf5_files = []
    # Load all .hdf5 files from the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.hdf5'):
            filepath = os.path.join(directory_path, filename)
            try:
                hdf5_file = h5py.File(filepath, 'r')  # Open file as hdf5
                hdf5_files.append(hdf5_file)
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
    
    # Preprocess the data
    preprocessed_data = {
        "rgb": preprocess_rgb_images(hdf5_files, image_size),
        "segmentation": preprocess_segmentation_masks(hdf5_files, image_size),
        "controls": preprocess_controls_multiple_episodes(hdf5_files),
        "frames": preprocess_frames_multiple_episodes(hdf5_files),
        "hlc": preprocess_hlc_multiple_episodes(hdf5_files),
        "light": preprocess_light_multiple_episodes(hdf5_files),
        "measurements": preprocess_measurements_multiple_episodes(hdf5_files),
    }
    
    return preprocessed_data


def preprocess_rgb_images(hdf5_files, image_size=(128, 128)):
    """
    Preprocess multiple HDF5 files containing RGB images by normalizing and resizing them.

    Parameters:
    - hdf5_files: List of HDF5 files containing the 'rgb' dataset.
    - image_size: Target size for resizing the images.

    Returns:
    - rgb_images_combined: Combined and preprocessed RGB images from all episodes.
    """
    rgb_images_combined = []

    # Loop through each HDF5 file and process the RGB images
    for hdf5_file in hdf5_files:
        if 'rgb' in hdf5_file:
            rgb_images = hdf5_file['rgb'][:]  # Access the 'rgb' dataset

            # Normalize RGB images
            rgb_images = rgb_images.astype(np.float32) / 255.0

            # Resize images
            rgb_images_resized = np.array([cv2.resize(img, image_size) for img in rgb_images])

            # Append to combined list
            rgb_images_combined.append(rgb_images_resized)
        else:
            print(f"Warning: 'rgb' dataset not found in {hdf5_file.filename}")

    # Concatenate all episodes into a single dataset
    if rgb_images_combined:
        rgb_images_combined = np.concatenate(rgb_images_combined, axis=0)
    else:
        rgb_images_combined = np.array([])  # Handle the case where no data was found

    return rgb_images_combined
def preprocess_segmentation_masks(hdf5_files, image_size=(128, 128)):
    """
    Preprocess multiple HDF5 files containing segmentation masks by resizing them.

    Parameters:
    - hdf5_files: List of HDF5 files containing the 'segmentation' dataset.
    - image_size: Target size for resizing the masks.

    Returns:
    - segmentation_masks_combined: Combined and preprocessed segmentation masks from all episodes.
    """
    segmentation_masks_combined = []

    # Loop through each HDF5 file and process the segmentation masks
    for hdf5_file in hdf5_files:
        segmentation_masks = hdf5_file['segmentation'][:]

        # Resize masks
        segmentation_masks_resized = np.array([cv2.resize(mask, image_size) for mask in segmentation_masks])

        # Append to combined list
        segmentation_masks_combined.append(segmentation_masks_resized)

    # Concatenate all episodes into a single dataset
    segmentation_masks_combined = np.concatenate(segmentation_masks_combined, axis=0)

    return segmentation_masks_combined


def preprocess_controls_multiple_episodes(hdf5_files):
    """
    Preprocess control data from multiple HDF5 files by concatenating and normalizing the controls.

    Parameters:
    - hdf5_files: List of HDF5 files containing the 'controls' dataset.

    Returns:
    - controls_normalized: Combined and normalized control data from all episodes.
    """
    controls_combined = []

    # Loop through each HDF5 file and concatenate the controls data
    for hdf5_file in hdf5_files:
        controls = hdf5_file['controls'][:]
        controls_combined.append(controls)

    # Concatenate all episodes' controls data
    controls_combined = np.concatenate(controls_combined, axis=0)

    # Normalize controls based on the max absolute value per feature
    controls_normalized = controls_combined / np.max(np.abs(controls_combined), axis=0)

    return controls_normalized


def preprocess_frames_multiple_episodes(hdf5_files):
    """
    Preprocess frame data from multiple HDF5 files by concatenating the frames.

    Parameters:
    - hdf5_files: List of HDF5 files containing the 'frame' dataset.

    Returns:
    - frames_combined: Combined frame data from all episodes.
    """
    frames_combined = []

    for hdf5_file in hdf5_files:
        frames = hdf5_file['frame'][:]
        frames_combined.append(frames)

    frames_combined = np.concatenate(frames_combined, axis=0)
    return frames_combined


def preprocess_hlc_multiple_episodes(hdf5_files):
    """
    Preprocess high-level command (hlc) data from multiple HDF5 files by concatenating the hlc data.

    Parameters:
    - hdf5_files: List of HDF5 files containing the 'hlc' dataset.

    Returns:
    - hlc_combined: Combined hlc data from all episodes.
    """
    hlc_combined = []

    for hdf5_file in hdf5_files:
        hlc = hdf5_file['hlc'][:]
        hlc_combined.append(hlc)

    hlc_combined = np.concatenate(hlc_combined, axis=0)
    return hlc_combined


def preprocess_light_multiple_episodes(hdf5_files):
    """
    Preprocess light data from multiple HDF5 files by concatenating the light data.

    Parameters:
    - hdf5_files: List of HDF5 files containing the 'light' dataset.

    Returns:
    - light_combined: Combined light data from all episodes.
    """
    light_combined = []

    for hdf5_file in hdf5_files:
        light = hdf5_file['light'][:]
        light_combined.append(light)

    light_combined = np.concatenate(light_combined, axis=0)
    return light_combined


def preprocess_measurements_multiple_episodes(hdf5_files):
    """
    Preprocess measurement data from multiple HDF5 files by concatenating the measurements.

    Parameters:
    - hdf5_files: List of HDF5 files containing the 'measurements' dataset.

    Returns:
    - measurements_combined: Combined measurement data from all episodes.
    """
    measurements_combined = []

    for hdf5_file in hdf5_files:
        measurements = hdf5_file['measurements'][:]
        measurements_combined.append(measurements)

    measurements_combined = np.concatenate(measurements_combined, axis=0)
    return measurements_combined


import numpy as np

def split_data_for_models_multiple_types(data_dict, num_models, data_distribution, val_split=0.2, test_split=0.1, data_types=None, random_seed=None):
    """
    Split the preprocessed data into subsets for local models, validation set, and test set.

    Parameters:
    - data_dict: Dictionary containing various preprocessed data types (X, y, etc.).
    - num_models: Number of local models to be implemented.
    - data_distribution: List of proportions for each model's training data.
    - val_split: Proportion of data to be used for validation.
    - test_split: Proportion of data to be used for testing.
    - data_types: List of data types to include in the split. If None, all data types are used.
    - random_seed: Seed for reproducibility of data splits.

    Returns:
    - model_data: Dictionary containing training data for each model.
    - val_data: Dictionary containing validation data.
    - test_data: Dictionary containing test data.
    """
    
    # Validate inputs
    if len(data_distribution) != num_models:
        raise ValueError("Length of data_distribution must match num_models.")
    if not np.isclose(np.sum(data_distribution), 1.0):
        raise ValueError("Data distribution must sum to 1.")

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    total_samples = len(data_dict['rgb'])  # Assumes all data types have the same length

    # Split indices into training, validation, and test sets
    indices = np.arange(total_samples)
    np.random.shuffle(indices)  # Shuffle indices for randomness

    # Calculate the number of samples for validation and test
    val_size = int(total_samples * val_split)
    test_size = int(total_samples * test_split)

    val_indices = indices[:val_size]
    test_indices = indices[val_size:val_size + test_size]
    train_indices = indices[val_size + test_size:]

    # Prepare validation and test data
    val_data = {
        key: np.array([data_dict[key][i] for i in val_indices])
        for key in (data_types or data_dict.keys()) if key in data_dict
    }

    test_data = {
        key: np.array([data_dict[key][i] for i in test_indices])
        for key in (data_types or data_dict.keys()) if key in data_dict
    }

    # Prepare model data for training
    model_data = {}
    total_train_samples = len(train_indices)

    start_idx = 0
    for model_id in range(num_models):
        # Calculate the number of samples for the current model
        num_samples = int(data_distribution[model_id] * total_train_samples)
        if model_id == num_models - 1:  # Last model gets any leftover data
            num_samples = total_train_samples - start_idx

        model_data[f'model_{model_id + 1}'] = {
            key: np.array([data_dict[key][train_indices[i]] for i in range(start_idx, start_idx + num_samples)])
            for key in (data_types or data_dict.keys()) if key in data_dict
        }

        start_idx += num_samples

    return model_data, val_data, test_data
