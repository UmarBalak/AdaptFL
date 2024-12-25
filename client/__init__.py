import logging
import os
from model.model_architecture import build_model
from model.model_training import main as train_main
from preprocess.preprocessing import preprocess_client_data

def setup_logger(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "client.log"),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def preprocess_data(client_id, raw_data_path, preprocessed_data_path, image_size=(128, 128)):
    """Preprocess raw data for the client."""
    try:
        preprocess_client_data(raw_data_path, preprocessed_data_path, client_id, image_size)
        logging.info(f"Preprocessing completed successfully for {client_id}")
    except Exception as e:
        logging.error(f"Error in preprocessing for {client_id}: {e}")

def train(client_id, data_path, save_dir, epochs=3, batch_size=32):
    """Train the model for the client."""
    try:
        train_main(client_id, data_path, save_dir, build_model, epochs, batch_size)
        logging.info(f"Training completed successfully for {client_id}")
    except Exception as e:
        logging.error(f"Error during training for {client_id}: {e}")

def main():
    """Main function to run preprocessing and training."""
    client_id = "1"
    # "/path/to/raw/data"
    raw_data_path = f"D:\AdaptFL\client{client_id}\data\episodes.hdf5"
    # "/path/to/preprocessed/data"
    preprocessed_data_path = f"D:\AdaptFL\client{client_id}\preprocessed_data"
    # "/path/to/save/models"
    save_dir = f"D:\AdaptFL"
    # "/path/to/logs"
    log_dir = f"D:\AdaptFL\client{client_id}\logs"

    setup_logger(log_dir)
    logging.info("Starting client pipeline")

    # preprocess_data(client_id, raw_data_path, preprocessed_data_path)
    train(client_id, preprocessed_data_path, save_dir)

    logging.info("Client pipeline completed")

if __name__ == "__main__":
    main()