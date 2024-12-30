import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
CSN = os.getenv("CSN")
script_directory = os.path.dirname(os.path.realpath(__file__))
CLIENT_CREDENTIALS_FILE = os.path.join(script_directory, "client_credentials.json")

def get_client_details(csn):
    url = f"{SERVER_URL}/client-details"
    response = requests.post(url, json={"csn": csn})

    if response.status_code == 200:
        client_details = response.json()
        # Store details in a local JSON file
        try:
            with open(CLIENT_CREDENTIALS_FILE, "w") as f:
                json.dump(client_details["data"], f, indent=4)
            print("Client details saved successfully.")
        except:
            print("Error during saving credentials.")
    else:
        print(f"Error: {response.json()['detail']}")

if __name__ == "__main__":
    get_client_details(CSN)