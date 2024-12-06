import requests

# URL of the global server's endpoint
server_url = "http://localhost:8000/download_weights"

# Download the global weights
response = requests.get(server_url)

if response.status_code == 200:
    with open("global_weights.npy", "wb") as f:
        f.write(response.content)
    print("Global weights downloaded successfully!")
else:
    print(f"Failed to download weights. Status Code: {response.status_code}")
    print(response.text)
