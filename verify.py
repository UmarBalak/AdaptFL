import os
import h5py

def load_hdf5_files_from_directory(directory_path):
    """
    Load HDF5 files from a specified directory.
    """
    hdf5_files = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".hdf5"):
            print(file_name)
            file_path = os.path.join(directory_path, file_name)
            with h5py.File(file_path, 'r') as hdf5_file:
                hdf5_files.append(hdf5_file)
    return hdf5_files

EPISODE_PATH = "data/episodes/"  # Update to the correct folder containing .hdf5 files
hdf5_files = load_hdf5_files_from_directory(EPISODE_PATH)
print("Data loaded successfully.")