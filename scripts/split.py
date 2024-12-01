import numpy as np

def load_processed_data(base_dir="data/processed/"):
    """
    Load processed data stored in .npy files into variables.

    Parameters:
    - base_dir: Base directory where processed data is stored.

    Returns:
    - Loaded data as NumPy arrays for splitting.
    """
    # Load RGB images
    rgb_images = np.load(f"{base_dir}/rgb/rgb_images.npy")

    # Load segmentation masks
    segmentation_masks = np.load(f"{base_dir}/segmentation/segmentation_masks.npy")

    # Load controls
    controls = np.load(f"{base_dir}/controls/controls.npy")

    # Load metadata
    frame = np.load(f"{base_dir}/metadata/frame.npy")
    hlc = np.load(f"{base_dir}/metadata/hlc.npy")
    light = np.load(f"{base_dir}/metadata/light.npy")
    measurements = np.load(f"{base_dir}/metadata/measurements.npy")

    return rgb_images, segmentation_masks, controls, frame, hlc, light, measurements

# Load data
rgb_images, segmentation_masks, controls, frame, hlc, light, measurements = load_processed_data(base_dir="data/processed/")

# Confirm shapes
print("Shapes of the loaded processed data:")
print(f"RGB Images: {rgb_images.shape}")
print(f"Segmentation Masks: {segmentation_masks.shape}")
print(f"Controls: {controls.shape}")
print(f"Frame: {frame.shape}")
print(f"HLC: {hlc.shape}")
print(f"Light: {light.shape}")
print(f"Measurements: {measurements.shape}")

import os
import numpy as np

import os
import h5py

def split_and_save_hdf5(rgb_images, segmentation_masks, controls, frames, hlc, light, measurements, 
                        save_dir, split_name, split_indices):
    """
    Save the split datasets into HDF5 format.

    Parameters:
    - rgb_images, segmentation_masks, controls, frames, hlc, light, measurements: Processed datasets.
    - save_dir: Directory to save the HDF5 files.
    - split_name: Name of the split (e.g., train, val, test).
    - split_indices: Indices for this split.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    # Create an HDF5 file for this split
    hdf5_path = os.path.join(save_dir, f"{split_name}.h5")
    with h5py.File(hdf5_path, "w") as hdf5_file:
        # Save RGB images
        hdf5_file.create_dataset("rgb_images", data=rgb_images[split_indices], compression="gzip")
        # Save segmentation masks
        hdf5_file.create_dataset("segmentation_masks", data=segmentation_masks[split_indices], compression="gzip")
        # Save controls
        hdf5_file.create_dataset("controls", data=controls[split_indices], compression="gzip")
        # Save frames
        hdf5_file.create_dataset("frames", data=frames[split_indices], compression="gzip")
        # Save high-level commands (HLC)
        hdf5_file.create_dataset("hlc", data=hlc[split_indices], compression="gzip")
        # Save traffic light statuses
        hdf5_file.create_dataset("light", data=light[split_indices], compression="gzip")
        # Save measurements
        hdf5_file.create_dataset("measurements", data=measurements[split_indices], compression="gzip")

    print(f"Data successfully saved to {hdf5_path}")


from sklearn.model_selection import train_test_split
import numpy as np

# Assuming rgb_images is a numpy array or any of the dataset arrays
dataset_size = len(rgb_images)  # Total number of samples in the dataset

# Create a shuffled array of indices
all_indices = np.arange(dataset_size)

# Split into training (70%), validation (15%), and test (15%)
train_indices, temp_indices = train_test_split(all_indices, test_size=0.3, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}, Test size: {len(test_indices)}")


# Split and save the data
split_and_save_hdf5(
    rgb_images=rgb_images,
    segmentation_masks=segmentation_masks,
    controls=controls,
    frames=frame,
    hlc=hlc,
    light=light,
    measurements=measurements,
    save_dir="processed_data",
    split_name="train",  # Can be 'train', 'val', or 'test'
    split_indices=train_indices
)

split_and_save_hdf5(
    rgb_images=rgb_images,
    segmentation_masks=segmentation_masks,
    controls=controls,
    frames=frame,
    hlc=hlc,
    light=light,
    measurements=measurements,
    save_dir="processed_data",
    split_name="val",
    split_indices=val_indices
)

split_and_save_hdf5(
    rgb_images=rgb_images,
    segmentation_masks=segmentation_masks,
    controls=controls,
    frames=frame,
    hlc=hlc,
    light=light,
    measurements=measurements,
    save_dir="processed_data",
    split_name="test",
    split_indices=test_indices
)


