import os
import numpy as np
import requests
import tensorflow as tf

# Define the URL of the global server's endpoint
server_url = "http://localhost:8000/upload_weights"  # Update with the correct URL

# Define the client directory and find the latest model file
client_id = "client1"  # Update this as needed
client_model_dir = f"D:/ADAPTFL/ADAPTFL_PROJECT/CLIENT1/models/local"  # Update with the correct local models directory

# Function to get the latest model file based on the timestamp in the file name
def get_latest_model(model_dir):
    try:
        # List all files in the directory and filter out only .h5 files
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
        
        if not model_files:
            print(f"No model files found in directory: {model_dir}")
            return None

        # Sort the files by the timestamp in the filename (format: client3_trained_model_YYYYMMDD_HHMMSS.h5)
        model_files.sort(reverse=True)  # Sort in descending order to get the latest model
        latest_model_file = model_files[0]
        latest_model_path = os.path.join(model_dir, latest_model_file)

        print(f"Latest model file found: {latest_model_path}")
        return latest_model_path
    except Exception as e:
        print(f"Error retrieving the latest model: {e}")
        return None

# Get the path to the latest model file
model_weights_path = get_latest_model(client_model_dir)

if model_weights_path:
    try:
        # Load the model and extract its weights
        model = tf.keras.models.load_model(model_weights_path)
        model_weights = model.get_weights()

        # Save weights as a numpy file (.npy) to upload
        weights_file_path = model_weights_path.replace(".h5", "_weights.npy")
        np.save(weights_file_path, model_weights)

        # Open the weights file to send as part of the request
        with open(weights_file_path, 'rb') as file:
            # Send the POST request with client_id as a query parameter
            response = requests.post(server_url, params={"client_id": client_id}, files={'file': file})

        # Check the response status code
        if response.status_code == 200:
            print("Model weights uploaded successfully!")
        else:
            print(f"Failed to upload model weights. Status Code: {response.status_code}")
            print(response.text)

    except FileNotFoundError:
        print(f"Error: The model weights file '{model_weights_path}' was not found.")
    except requests.exceptions.RequestException as e:
        print(f"Error during the request: {str(e)}")
else:
    print("No valid model found to upload.")
