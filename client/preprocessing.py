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
