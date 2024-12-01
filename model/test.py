import requests
import h5py
import io

# Load HDF5 file from a URL
def load_hdf5_from_url(url, number_of_episodes=1):
    hdf5_files = []
    for episode_number in range(number_of_episodes):
      new_url = url + f'episode_{episode_number}.hdf5'
      response = requests.get(new_url)
      response.raise_for_status()  # Raise an error for bad responses
      hdf5_files.append(h5py.File(io.BytesIO(response.content), 'r'))
      print(f"episode_{episode_number}.hdf5 loaded successfully.")
    return hdf5_files

# hdf5_url = 'https://huggingface.co/datasets/nightmare-nectarine/segmentation-carla-driving/resolve/main/train/'
# hdf5_files = load_hdf5_from_url(hdf5_url, 3)

import h5py
import os
import random

def load_random_hdf5_from_folder(folder_path, number_of_episodes=1):
    # Get all .hdf5 files in the folder
    available_files = [f for f in os.listdir(folder_path) if f.endswith('.hdf5')]
    
    # If there are fewer available files than requested, load all
    if len(available_files) < number_of_episodes:
        print(f"Only {len(available_files)} files found. Loading all available files.")
        selected_files = available_files
    else:
        # Randomly select the specified number of episodes
        selected_files = random.sample(available_files, number_of_episodes)
    
    hdf5_files = []
    for file_name in selected_files:
        file_path = os.path.join(folder_path, file_name)
        hdf5_files.append(h5py.File(file_path, 'r'))
        print(f"{file_name} loaded successfully.")
    
    return hdf5_files

folder_path = 'E:\Major-Project\Dataset'
hdf5_files = load_random_hdf5_from_folder(folder_path, 14)


# Explore the HDF5 file structure to see what datasets are present
def explore_hdf5_structure(hdf5_file):
    """
    Print the structure and contents of an HDF5 file to explore available datasets.
    """
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")

    hdf5_file.visititems(print_structure)

# Check each episode file's structure
# for episode_file in hdf5_files:
#     print("Exploring HDF5 structure:")
#     explore_hdf5_structure(episode_file)


import numpy as np
import cv2

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
        rgb_images = hdf5_file['rgb'][:]

        # Normalize RGB images
        rgb_images = rgb_images.astype(np.float32) / 255.0

        # Resize images
        rgb_images_resized = np.array([cv2.resize(img, image_size) for img in rgb_images])

        # Append to combined list
        rgb_images_combined.append(rgb_images_resized)

    # Concatenate all episodes into a single dataset
    rgb_images_combined = np.concatenate(rgb_images_combined, axis=0)

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

import numpy as np

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


# Call the functions
rgb_data = preprocess_rgb_images(hdf5_files)
segmentation_data = preprocess_segmentation_masks(hdf5_files)
controls_normalized = preprocess_controls_multiple_episodes(hdf5_files)
frame_data = preprocess_frames_multiple_episodes(hdf5_files)
hlc_data = preprocess_hlc_multiple_episodes(hdf5_files)
light_data = preprocess_light_multiple_episodes(hdf5_files)
measurements_data = preprocess_measurements_multiple_episodes(hdf5_files)

# Output shapes for verification
print(f"RGB images shape: {rgb_data.shape}")
print(f"Segmentation masks shape: {segmentation_data.shape}")
print(f"Controls shape: {controls_normalized.shape}")
print(f"Frames shape: {frame_data.shape}")
print(f"HLC shape: {hlc_data.shape}")
print(f"Light shape: {light_data.shape}")
print(f"Measurements shape: {measurements_data.shape}")


def split_data_for_models_multiple_types(data_dict, num_models, data_distribution, val_split=0.2, data_types=None, random_seed=None):
    """
    Split the preprocessed data into subsets for local models and validation set.

    Parameters:
    - data_dict: Dictionary containing various preprocessed data types (X, y, etc.).
    - num_models: Number of local models to be implemented.
    - data_distribution: List of proportions for each model's training data.
    - val_split: Proportion of data to be used for validation.
    - data_types: List of data types to include in the split. If None, all data types are used.
    - random_seed: Seed for reproducibility of data splits.

    Returns:
    - model_data: Dictionary containing training data for each model.
    - val_data: Dictionary containing validation data.
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

    # Split indices into training and validation
    indices = np.arange(total_samples)
    np.random.shuffle(indices)  # Shuffle indices for randomness

    # Calculate the number of samples for validation
    val_size = int(total_samples * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Prepare validation data
    val_data = {
        key: np.array([data_dict[key][i] for i in val_indices])
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

    return model_data, val_data


data_dict = {
    'controls': controls_normalized,
    'frame': frame_data,
    'hlc': hlc_data,
    'light': light_data,
    'measurements': measurements_data,
    'rgb': rgb_data,
    'segmentation': segmentation_data,
}

# Specify the number of local models, data distribution, and types to include
num_local_models = 3
data_distribution = [0.4, 0.3, 0.3]
data_types_to_include = ['rgb', 'segmentation', 'hlc', 'light', 'measurements', 'controls', 'frame']

# Split the data for selected types
model_data, val_data = split_data_for_models_multiple_types(data_dict, num_local_models, data_distribution=data_distribution, val_split=0.2)

# model_data = split_data_for_models_multiple_types(data_dict, num_local_models, data_distribution, data_types=data_types_to_include, random_seed=42)

# Output shapes for verification
for model_id, data in model_data.items():
    print(f"{model_id} - Data shapes:")
    for key, value in data.items():
        print(f"  {key}: {value.shape}")


import tensorflow as tf
import keras
from keras import layers, Model, Input

# Defining the model architecture
# def create_model(input_shapes):
#     """
#     Create a multi-input model based on the shapes of the input data.

#     Parameters:
#     - input_shapes: Dictionary containing the shape of each input type (e.g., 'rgb', 'segmentation', etc.)

#     Returns:
#     - A compiled Keras model with multiple inputs for multi-class classification.
#     """
#     # Input layers for each type of data
#     rgb_input = Input(shape=input_shapes['rgb'], name='rgb_input')
#     seg_input = Input(shape=input_shapes['segmentation'], name='seg_input')
#     controls_input = Input(shape=input_shapes['controls'], name='controls_input')
#     light_input = Input(shape=input_shapes['light'], name='light_input')
#     measurements_input = Input(shape=input_shapes['measurements'], name='measurements_input')

#     # Simple CNN for RGB and Segmentation inputs
#     rgb_features = layers.Conv2D(32, (3, 3), activation='relu')(rgb_input)
#     rgb_features = layers.MaxPooling2D((2, 2))(rgb_features)
#     rgb_features = layers.Flatten()(rgb_features)

#     seg_features = layers.Conv2D(32, (3, 3), activation='relu')(seg_input)
#     seg_features = layers.MaxPooling2D((2, 2))(seg_features)
#     seg_features = layers.Flatten()(seg_features)

#     # Dense layers for other inputs
#     controls_features = layers.Dense(16, activation='relu')(controls_input)
#     light_features = layers.Dense(16, activation='relu')(light_input)
#     measurements_features = layers.Dense(16, activation='relu')(measurements_input)

#     # Concatenate all features
#     concatenated = layers.concatenate([rgb_features, seg_features, controls_features, light_features, measurements_features])

#     # Output layer (adjust depending on your task, e.g., classification, regression)
#     output = layers.Dense(4, activation='softmax')(concatenated)

#     # Create model
#     model = Model(inputs=[rgb_input, seg_input, controls_input, light_input, measurements_input], outputs=output)

#     # Compile model
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     return model

from keras import layers, Model
from keras.layers import Input
from keras.applications import VGG16  # Example for using a pre-trained model
from keras.layers import Dropout, BatchNormalization, GlobalAveragePooling2D

def create_model(input_shapes):
    """
    Create a more accurate multi-input model based on the shapes of the input data.

    Parameters:
    - input_shapes: Dictionary containing the shape of each input type (e.g., 'rgb', 'segmentation', etc.)

    Returns:
    - A compiled Keras model with multiple inputs for multi-class classification.
    """
    # Input layers for each type of data
    rgb_input = Input(shape=input_shapes['rgb'], name='rgb_input')
    seg_input = Input(shape=input_shapes['segmentation'], name='seg_input')
    controls_input = Input(shape=input_shapes['controls'], name='controls_input')
    light_input = Input(shape=input_shapes['light'], name='light_input')
    measurements_input = Input(shape=input_shapes['measurements'], name='measurements_input')

    # Using a pre-trained model for RGB input
    base_model_rgb = VGG16(weights='imagenet', include_top=False, input_tensor=rgb_input)
    rgb_features = base_model_rgb.output
    rgb_features = GlobalAveragePooling2D()(rgb_features)
    rgb_features = Dropout(0.5)(rgb_features)  # Adding Dropout

    # CNN for Segmentation input
    seg_features = layers.Conv2D(32, (3, 3), activation='relu')(seg_input)
    seg_features = layers.MaxPooling2D((2, 2))(seg_features)
    seg_features = layers.Conv2D(64, (3, 3), activation='relu')(seg_features)
    seg_features = layers.MaxPooling2D((2, 2))(seg_features)
    seg_features = layers.Flatten()(seg_features)
    seg_features = Dropout(0.5)(seg_features)  # Adding Dropout

    # Dense layers for other inputs with Batch Normalization
    controls_features = layers.Dense(32, activation='relu')(controls_input)
    controls_features = BatchNormalization()(controls_features)
    light_features = layers.Dense(32, activation='relu')(light_input)
    light_features = BatchNormalization()(light_features)
    measurements_features = layers.Dense(32, activation='relu')(measurements_input)
    measurements_features = BatchNormalization()(measurements_features)

    # Concatenate all features
    concatenated = layers.concatenate([rgb_features, seg_features, controls_features, light_features, measurements_features])

    # Output layer (adjust depending on your task, e.g., classification, regression)
    output = layers.Dense(4, activation='softmax')(concatenated)

    # Create model
    model = Model(inputs=[rgb_input, seg_input, controls_input, light_input, measurements_input], outputs=output)

    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


local_models = []
for model_id, data in model_data.items():
    print(f"Training {model_id}...")

    # Get the shapes of the inputs
    input_shapes = {key: data[key].shape[1:] for key in data_types_to_include}

    # Create a model for this data
    model = create_model(input_shapes)

    labels = data['hlc']

    model.fit(
        [data['rgb'], data['segmentation'], data['controls'], data['light'], data['measurements']],
        labels, epochs=3, batch_size=32,
        validation_data=(
                [val_data['rgb'], val_data['segmentation'], val_data['controls'], val_data['light'], val_data['measurements']],
                val_data['hlc']
            ),
    )

    local_models.append(model)
    print(f"{model_id} training complete.\n")

# for id, model in enumerate(local_models):
#     filepath = f"local_model_{id}_weights.weights.h5"
#     model.save_weights(filepath)
#     print(f"{id} training complete and weights saved as {filepath}.\n")

#     print(f"\nWeights for {model_id}:")
#     for layer in model.layers:
#         print(f"Layer: {layer.name}, Weights: {layer.get_weights()}")

