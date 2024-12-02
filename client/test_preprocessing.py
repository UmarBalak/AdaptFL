from preprocessing import *

def test_preprocessing():
    # Load HDF5 files from the specified directory
    EPISODE_PATH = "data/episodes/"  # Update to the correct folder containing .hdf5 files
    preprocessed_data = load_and_preprocess_dynamic_episodes(EPISODE_PATH)
    
    # Run all preprocessing functions
    rgb_data = preprocessed_data['rgb']
    segmentation_data = preprocessed_data['segmentation']
    controls_normalized = preprocessed_data['controls']
    frame_data = preprocessed_data['frames']
    hlc_data = preprocessed_data['hlc']
    light_data = preprocessed_data['light']
    measurements_data = preprocessed_data['measurements']

    # Output shapes for verification
    print(f"RGB images shape: {rgb_data.shape}")
    print(f"Segmentation masks shape: {segmentation_data.shape}")
    print(f"Controls shape: {controls_normalized.shape}")
    print(f"Frames shape: {frame_data.shape}")
    print(f"HLC shape: {hlc_data.shape}")
    print(f"Light shape: {light_data.shape}")
    print(f"Measurements shape: {measurements_data.shape}")

if __name__ == "__main__":
    test_preprocessing()
