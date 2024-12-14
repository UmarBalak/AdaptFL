import os
import numpy as np

def load_preprocessed_data(preprocessed_data_path):
    """
    Load the preprocessed data from a `.npz` file.
    """
    if not os.path.exists(preprocessed_data_path):
        raise FileNotFoundError(f"Preprocessed data file {preprocessed_data_path} does not exist.")
    
    data = np.load(preprocessed_data_path, allow_pickle=True)
    
    return {
        "rgb": data["rgb"],
        "segmentation": data["segmentation"],
        "controls": data["controls"],
        "frames": data["frames"],
        "hlc": data["hlc"],
        "light": data["light"],
        "measurements": data["measurements"]
    }
