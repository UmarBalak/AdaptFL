import h5py
import numpy as np
import os
import cv2
import logging
import sys  # To capture command-line arguments
import logging.handlers

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
def setup_logger(client_id):
    log_file = f"../AdaptFL_Project/{client_id}/logs/preprocessing.log"
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=3
    )
    logging.basicConfig(
        handlers=[handler],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
def preprocess_client_data(client_id, image_size=(128, 128)):
    """
    Preprocess data for a specific client.

    Parameters:
    - client_id: Identifier for the client (e.g., client1).
    - image_size: Target size for resizing images.

    Saves:
    - Preprocessed data into the `preprocessed_data` folder of the respective client.
    """
    setup_logger(client_id)
    logging.info(f"Starting preprocessing for {client_id}")

    try:
        # Paths
        data_path = f"../AdaptFL_Project/{client_id}/data"
        preprocessed_path = f"../AdaptFL_Project/{client_id}/preprocessed_data"
        os.makedirs(preprocessed_path, exist_ok=True)

        # Load and preprocess data
        hdf5_files = load_hdf5_files(data_path)
        preprocessed_data = {
            "rgb": preprocess_rgb_images(hdf5_files, image_size),
            "segmentation": preprocess_segmentation_masks(hdf5_files, image_size),
            "controls": preprocess_controls_multiple_episodes(hdf5_files),
            "frames": preprocess_frames_multiple_episodes(hdf5_files),
            "hlc": preprocess_hlc_multiple_episodes(hdf5_files),
            "light": preprocess_light_multiple_episodes(hdf5_files),
            "measurements": preprocess_measurements_multiple_episodes(hdf5_files),
        }

        # Save preprocessed data into a single file
        save_preprocessed_data(preprocessed_path, preprocessed_data)

        logging.info(f"Preprocessing completed successfully for {client_id}")

    except Exception as e:
        logging.error(f"Error in preprocessing for {client_id}: {e}")
def load_hdf5_files(directory_path):
    """
    Load all HDF5 files from a directory.

    Parameters:
    - directory_path: Path to the directory containing the HDF5 files.

    Returns:
    - hdf5_files: List of opened HDF5 files.
    """
    hdf5_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".hdf5"):
            filepath = os.path.join(directory_path, filename)
            try:
                hdf5_file = h5py.File(filepath, "r")
                hdf5_files.append(hdf5_file)
            except Exception as e:
                logging.warning(f"Error loading file {filename}: {e}")
    return hdf5_files
def save_preprocessed_data(save_path, data_dict):
    """
    Save preprocessed data into a single .npz file containing all datasets.

    Parameters:
    - save_path: Directory to save the data.
    - data_dict: Dictionary containing dataset names as keys and preprocessed data as values.
    """
    save_file = os.path.join(save_path, "preprocessed_data.npz")

    # Save all datasets into a single .npz file
    np.savez_compressed(save_file, **data_dict)
    logging.info(f"Saved preprocessed data into: {save_file}")

if __name__ == "__main__":
    preprocess_client_data("client1")