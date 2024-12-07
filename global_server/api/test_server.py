import requests

# Define the server URL
BASE_URL = "http://127.0.0.1:8000"  # Local server address

# Function to upload client weights
def upload_weights(client_id, file_path):
    url = f"{BASE_URL}/upload_weights?client_id={client_id}"
    with open(file_path, 'rb') as file:
        files = {'file': (file_path, file)}
        response = requests.post(url, files=files)
        print(f"Upload Status for {client_id}: {response.status_code}")
        print(response.text)

# Function to aggregate client weights
def aggregate_weights():
    url = f"{BASE_URL}/aggregate_weights"
    response = requests.post(url)
    print(f"Aggregation Status: {response.status_code}")
    print(response.text)

# Function to download global model
def download_weights():
    url = f"{BASE_URL}/download_weights"
    response = requests.get(url)
    print(f"Download Status: {response.status_code}")
    if response.status_code == 200:
        with open("global_model.h5", 'wb') as f:
            f.write(response.content)
            print("Global model saved as 'global_model.h5'")
    else:
        print(response.text)

# Testing the flow
if __name__ == "__main__":
    # Test upload
    upload_weights("client1", "../AdaptFL_Project\client1\models\local\client1_trained_model_20241206_163229.h5")
    upload_weights("client2", "../AdaptFL_Project\client2\models\local\client2_trained_model_20241206_163304.h5")
    upload_weights("client3", "../AdaptFL_Project\client3\models\local\client3_trained_model_20241206_163343.h5")

    # Test aggregation
    aggregate_weights()

    # Test downloading the global model
    download_weights()
