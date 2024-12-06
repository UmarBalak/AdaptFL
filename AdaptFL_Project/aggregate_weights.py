# import requests

# # Define the URL of the global server's endpoint for aggregation
# aggregate_url = "http://localhost:8000/aggregate_weights"

# # Send the POST request to aggregate weights
# response = requests.post(aggregate_url)

# # Check the response
# if response.status_code == 200:
#     print("Global model updated with aggregated weights!")
# else:
#     print(f"Failed to aggregate weights. Status Code: {response.status_code}")
#     print(response.text)


import requests

# Define the URL of the global server's endpoint for aggregation
aggregate_url = "http://localhost:8000/aggregate_weights"

try:
    # Define paths for weights directory and global model path
    weights_dir = "../AdaptFL_Project/global_server/weights"  # Ensure this path is correct
    global_model_path = "../AdaptFL_Project/global_server/global_model/global_model.h5"  # Ensure this path is correct

    # Send the POST request to aggregate weights, passing the paths as query parameters
    response = requests.post(aggregate_url, params={"WEIGHTS_DIR": weights_dir, "GLOBAL_MODEL_PATH": global_model_path})


    # Check the response status code
    if response.status_code == 200:
        print("Global model updated with aggregated weights!")
    else:
        print(f"Failed to aggregate weights. Status Code: {response.status_code}")
        print(response.text)

except requests.exceptions.Timeout:
    print("Error: The request timed out. Please check the server.")
except requests.exceptions.RequestException as e:
    print(f"Error during the request: {str(e)}")
