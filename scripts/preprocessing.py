import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm

# Paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

# Image size for resizing
IMG_SIZE = (128, 128)

# Utility: Load all .hdf5 files
def load_hdf5_files(directory):
    """Load all HDF5 files in a directory."""
    files = []
    for file in os.listdir(directory):
        if file.endswith(".hdf5"):
            filepath = os.path.join(directory, file)
            files.append(h5py.File(filepath, 'r'))
    return files

# Preprocess RGB images
def preprocess_rgb_images(hdf5_files, output_dir, image_size=IMG_SIZE):
    os.makedirs(output_dir, exist_ok=True)
    rgb_images_combined = []

    for file in tqdm(hdf5_files, desc="Processing RGB Images"):
        rgb_images = file['rgb'][:]
        rgb_images = rgb_images.astype(np.float32) / 255.0  # Normalize
        rgb_images_resized = np.array([cv2.resize(img, image_size) for img in rgb_images])
        rgb_images_combined.append(rgb_images_resized)

    rgb_images_combined = np.concatenate(rgb_images_combined, axis=0)
    np.save(os.path.join(output_dir, "rgb_images.npy"), rgb_images_combined)
    return rgb_images_combined

# Preprocess segmentation masks
def preprocess_segmentation_masks(hdf5_files, output_dir, image_size=IMG_SIZE):
    os.makedirs(output_dir, exist_ok=True)
    segmentation_combined = []

    for file in tqdm(hdf5_files, desc="Processing Segmentation Masks"):
        segmentation_masks = file['segmentation'][:]
        segmentation_resized = np.array([cv2.resize(mask, image_size) for mask in segmentation_masks])
        segmentation_combined.append(segmentation_resized)

    segmentation_combined = np.concatenate(segmentation_combined, axis=0)
    np.save(os.path.join(output_dir, "segmentation_masks.npy"), segmentation_combined)
    return segmentation_combined

# Preprocess numerical data
def preprocess_controls(hdf5_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    controls_combined = []

    for file in tqdm(hdf5_files, desc="Processing Controls"):
        controls = file['controls'][:]
        controls_combined.append(controls)

    controls_combined = np.concatenate(controls_combined, axis=0)
    np.save(os.path.join(output_dir, "controls.npy"), controls_combined)
    return controls_combined

# Preprocess metadata (frame, hlc, light, measurements)
def preprocess_metadata(hdf5_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata_combined = {'frame': [], 'hlc': [], 'light': [], 'measurements': []}

    for file in tqdm(hdf5_files, desc="Processing Metadata"):
        metadata_combined['frame'].append(file['frame'][:])
        metadata_combined['hlc'].append(file['hlc'][:])
        metadata_combined['light'].append(file['light'][:])
        metadata_combined['measurements'].append(file['measurements'][:])

    # Concatenate metadata
    for key in metadata_combined:
        metadata_combined[key] = np.concatenate(metadata_combined[key], axis=0)
        np.save(os.path.join(output_dir, f"{key}.npy"), metadata_combined[key])

    return metadata_combined

# Main preprocessing function
def main():
    # Create processed data directory
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # Load all HDF5 files
    hdf5_files = load_hdf5_files(RAW_DATA_PATH)

    # Preprocess RGB images
    preprocess_rgb_images(hdf5_files, os.path.join(PROCESSED_DATA_PATH, "rgb"))

    # Preprocess segmentation masks
    preprocess_segmentation_masks(hdf5_files, os.path.join(PROCESSED_DATA_PATH, "segmentation"))

    # Preprocess controls
    preprocess_controls(hdf5_files, os.path.join(PROCESSED_DATA_PATH, "controls"))

    # Preprocess metadata
    preprocess_metadata(hdf5_files, os.path.join(PROCESSED_DATA_PATH, "metadata"))

    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
