import requests
import numpy as np
from loguru import logger
import os

# Server Configuration
BASE_URL = "http://127.0.0.1:5000"
MODEL_SAVE_PATH  = "global_server/temp/"

# Dummy weights for testing
weights_1 = np.random.rand(5, 5).tolist()
weights_2 = np.random.rand(5, 5).tolist()

# Send weights from client 1
response_1 = requests.post(f"{BASE_URL}/send_weights", json={"weights": weights_1})
print("Client 1 Response:", response_1.json())

# Send weights from client 2
response_2 = requests.post(f"{BASE_URL}/send_weights", json={"weights": weights_2})
print("Client 2 Response:", response_2.json())

# Retrieve aggregated weights
global_weights = requests.get(f"{BASE_URL}/get_weights/")
np.save(os.path.join(MODEL_SAVE_PATH, "global_weights.npy"), global_weights)
logger.info("Global weights saved to global_weights.npy.")
print("Global Weights:", global_weights.json())
