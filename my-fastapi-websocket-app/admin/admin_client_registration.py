import requests
import sqlite3
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Server URL and admin API key
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

# Database file
DB_FILE = "client_registration.db"

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            csn TEXT UNIQUE NOT NULL,
            client_id TEXT NOT NULL,
            api_key TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Save client credentials in SQLite
def save_to_db(csn, client_id, api_key):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO clients (csn, client_id, api_key) VALUES (?, ?, ?)
    """, (csn, client_id, api_key))
    conn.commit()
    conn.close()

# Register client with server
def register_client(csn):
    payload = {
        "csn": csn,
        "admin_api_key": ADMIN_API_KEY
    }
    try:
        response = requests.post(f"{SERVER_URL}/register", json=payload)
        if response.status_code == 200:
            data = response.json().get("data", {})
            client_id = data.get("client_id")
            api_key = data.get("api_key")
            if client_id and api_key:
                save_to_db(csn, client_id, api_key)
                print(f"Client registered successfully: CSN={csn}, Client ID={client_id}")
            else:
                print("Invalid response from server")
        else:
            print(f"Failed to register client: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the server: {e}")

# Generate a unique CSN
def generate_csn():
    return f"CSN-{uuid.uuid4()}"

if __name__ == "__main__":
    init_db()
    # Option to generate CSN automatically for testing
    auto_generate = input("Do you want to auto-generate a CSN? (y/n): ").strip().lower()
    if auto_generate == "y":
        csn = generate_csn()
        print(f"Generated CSN: {csn}")
    else:
        csn = input("Enter the CSN (Client Serial Number): ").strip()
    
    register_client(csn)
