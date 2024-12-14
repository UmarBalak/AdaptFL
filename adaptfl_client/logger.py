import logging
import os

def setup_logger(client_id: str, log_dir: str = "./logs"):
    """
    Sets up a logger for the client framework.

    Args:
        client_id (str): Unique identifier for the client.
        log_dir (str): Directory to save log files.
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Define log file path
    log_file = os.path.join(log_dir, f"{client_id}.log")

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Log to console as well
        ]
    )

    logging.info(f"Logger initialized for client {client_id}. Logs are saved to {log_file}.")
