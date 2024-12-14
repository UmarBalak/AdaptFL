import logging
from adaptfl_client.logger import setup_logger
from adaptfl_client.config_loader import ConfigLoader

def main():
    # Load configuration
    config_path = "adaptfl_client\config.json"  # Replace with dynamic path if necessary
    config = ConfigLoader(config_path)

    # Set up the logger
    client_id = config.get("client_id")
    log_dir = "./logs"
    setup_logger(client_id, log_dir)

    logging.info("Client framework initialized successfully.")

if __name__ == "__main__":
    main()
