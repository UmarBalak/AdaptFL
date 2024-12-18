import os
import logging
import numpy as np
import h5py
from skimage.transform import resize

# --- Utility functions ---

def setup_logger(client_id, log_dir):
    """
    Set up a client-specific logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"{client_id}_preprocessing.log"),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

# --- Preprocessing functions ---

def load_hdf5_file(file_path):
    """
    Load a single HDF5 file from the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The HDF5 file {file_path} does not exist.")
    
    return h5py.File(file_path, "r")

def preprocess_rgb_images(file, image_size):
    """
    Preprocess the RGB images from a single HDF5 file.
    Resize and normalize pixel values.
    """
    rgb_data = file["rgb"][:]
    resized_images = np.array([resize(img, image_size) for img in rgb_data])
    normalized_images = resized_images / 255.0
    return normalized_images

def preprocess_segmentation_masks(file, image_size):
    """
    Preprocess segmentation masks from a single HDF5 file.
    Resize masks to the target size.
    """
    seg_data = file["segmentation"][:]
    resized_masks = np.array([resize(mask, image_size) for mask in seg_data])
    return resized_masks

def preprocess_controls(file):
    """
    Preprocess control data from a single HDF5 file.
    """
    controls = file["controls"][:]
    return controls

def preprocess_frames(file):
    """
    Preprocess frame indices from a single HDF5 file.
    """
    frames = file["frame"][:]
    return frames

def preprocess_hlc(file):
    """
    Preprocess high-level commands (HLC) from a single HDF5 file.
    """
    hlc = file["hlc"][:]
    return hlc

def preprocess_light(file):
    """
    Preprocess traffic light status from a single HDF5 file.
    """
    light = file["light"][:]
    return light

def preprocess_measurements(file):
    """
    Preprocess speed measurements from a single HDF5 file.
    """
    measurements = file["measurements"][:]
    return measurements

def save_preprocessed_data(output_path, data):
    """
    Save preprocessed data to a NumPy file.
    """
    output_file = os.path.join(output_path, "preprocessed_data.npz")
    np.savez_compressed(output_file, **data)
    logging.info(f"Preprocessed data saved to {output_file}")

# --- Main preprocessing function ---

def preprocess_client_data(data_path, output_path, client_id, image_size=(128, 128), log_dir=None):
    """
    Preprocess data for the client identified by the client_id.

    Parameters:
    - data_path: Path to the raw data (single HDF5 file).
    - output_path: Path to store preprocessed data.
    - client_id: Client identifier for logging.
    - image_size: Target size for resizing images.
    - log_dir: Path to store log files.

    Saves:
    - Preprocessed data into the output_path folder for the respective client.
    """
    if log_dir is None:
        log_dir = os.path.join(output_path, "logs")

    setup_logger(client_id, log_dir)  # Set up client-specific logging
    logging.info(f"Starting preprocessing for {client_id}")

    try:
        # Load the single HDF5 file
        hdf5_file = load_hdf5_file(data_path)

        # Preprocess the data
        preprocessed_data = {
            "rgb": preprocess_rgb_images(hdf5_file, image_size),
            "segmentation": preprocess_segmentation_masks(hdf5_file, image_size),
            "controls": preprocess_controls(hdf5_file),
            "frames": preprocess_frames(hdf5_file),
            "hlc": preprocess_hlc(hdf5_file),
            "light": preprocess_light(hdf5_file),
            "measurements": preprocess_measurements(hdf5_file),
        }

        # Save preprocessed data into a single file
        save_preprocessed_data(output_path, preprocessed_data)

        logging.info(f"Preprocessing completed successfully for {client_id}")

    except Exception as e:
        logging.error(f"Error in preprocessing for {client_id}: {e}")
