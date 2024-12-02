from client import *

EPISODE_PATH = "data/episodes/"  # Update to the correct folder containing .hdf5 files
preprocessed_data = load_and_preprocess_dynamic_episodes(EPISODE_PATH)

# Combine all preprocessed data into a dictionary
data_dict = {
    'controls': preprocessed_data['controls'],
    'frames': preprocessed_data['frames'],
    'hlc': preprocessed_data['hlc'],
    'light': preprocessed_data['light'],
    'measurements': preprocessed_data['measurements'],
    'rgb': preprocessed_data['rgb'],
    'segmentation': preprocessed_data['segmentation'],
}

model_data, val_data, test_data = split_data_for_models_multiple_types(
    data_dict=data_dict,
    num_models=3,
    data_distribution=[0.4, 0.4, 0.2],
    val_split=0.2,
    test_split=0.1,
    data_types=None,
    random_seed=42
)

# Train each model on its local data
for model_id, model_train_data in model_data.items():
    print(f"Training {model_id}...")
    train_local_model(
        model_id=model_id,
        train_data=model_train_data,
        val_data=val_data,
        epochs=10,
        batch_size=32,
        save_dir="saved_models"
    )
