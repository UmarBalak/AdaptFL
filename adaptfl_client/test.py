from logger import setup_logger
import logging
import os
def test_logger():
    client_id = "test_client"
    log_dir = "./test_logs"

    # Setup logger
    setup_logger(client_id, log_dir)

    # Log some test messages
    logging.info("This is an INFO log.")
    logging.warning("This is a WARNING log.")
    logging.error("This is an ERROR log.")

    # Check the log file
    log_file = f"{log_dir}/{client_id}.log"
    assert os.path.exists(log_file), "Log file not created."
    print("Logger tests passed. Check the logs for messages.")

if __name__ == "__main__":
    test_logger()
