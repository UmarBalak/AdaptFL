import h5py
import numpy as np
import os
import cv2
import logging

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

    for hdf5_file in hdf5_files:
        if 'rgb' in hdf5_file:
            rgb_images = hdf5_file['rgb'][:]  # Access the 'rgb' dataset
            rgb_images = rgb_images.astype(np.float32) / 255.0  # Normalize RGB images
            rgb_images_resized = np.array([cv2.resize(img, image_size) for img in rgb_images])
            rgb_images_combined.append(rgb_images_resized)
        else:
            print(f"Warning: 'rgb' dataset not found in {hdf5_file.filename}")

    if rgb_images_combined:
        rgb_images_combined = np.concatenate(rgb_images_combined, axis=0)
    else:
        rgb_images_combined = np.array([])

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

    for hdf5_file in hdf5_files:
        segmentation_masks = hdf5_file['segmentation'][:]
        segmentation_masks_resized = np.array([cv2.resize(mask, image_size) for mask in segmentation_masks])
        segmentation_masks_combined.append(segmentation_masks_resized)

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

    for hdf5_file in hdf5_files:
        controls = hdf5_file['controls'][:]
        controls_combined.append(controls)

    controls_combined = np.concatenate(controls_combined, axis=0)
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

def setup_logger():
    """Setup logger for preprocessing."""
    log_file = "../AdaptFL_Project/test_data/logs/test_preprocessing.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def preprocess_test_data(image_size=(128, 128)):
    """
    Preprocess data for the test episode.

    Saves:
    - Preprocessed data into the `processeddata` folder.
    """
    setup_logger()
    logging.info("Starting preprocessing for test data")

    try:
        # Paths
        test_data_path = "../AdaptFL_Project/test_data"
        processed_data_path = "../AdaptFL_Project/test_data/processed_test_data"
        os.makedirs(processed_data_path, exist_ok=True)

        # Load and preprocess data
        test_episode_file = os.path.join(test_data_path, "test_episode.hdf5")
        hdf5_files = [h5py.File(test_episode_file, "r")]

        preprocessed_data = {
            "rgb": preprocess_rgb_images(hdf5_files, image_size),
            "segmentation": preprocess_segmentation_masks(hdf5_files, image_size),
            "controls": preprocess_controls_multiple_episodes(hdf5_files),
            "frames": preprocess_frames_multiple_episodes(hdf5_files),
            "hlc": preprocess_hlc_multiple_episodes(hdf5_files),
            "light": preprocess_light_multiple_episodes(hdf5_files),
            "measurements": preprocess_measurements_multiple_episodes(hdf5_files),
        }

        # Save preprocessed data
        for key, data in preprocessed_data.items():
            save_preprocessed_data(processed_data_path, key, data)

        logging.info("Preprocessing completed successfully for test data")

    except Exception as e:
        logging.error(f"Error in preprocessing test data: {e}")

def save_preprocessed_data(save_path, dataset_name, data):
    """
    Save preprocessed data to a file.

    Parameters:
    - save_path: Directory to save the data.
    - dataset_name: Name of the dataset (e.g., 'rgb').
    - data: Preprocessed data to save.
    """
    save_file = os.path.join(save_path, f"{dataset_name}.npy")
    np.save(save_file, data)
    logging.info(f"Saved preprocessed data: {save_file}")

if __name__ == "__main__":
    preprocess_test_data()
