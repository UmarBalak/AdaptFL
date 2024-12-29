from azure.storage.blob import BlobServiceClient, generate_container_sas, ContainerSasPermissions
from datetime import datetime, timedelta
import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()



def generate_sas_token(duration_days: int = 90) -> str:
    """
    Generate a SAS token for container access.
    
    Args:
        duration_days: Number of days the token should be valid
        
    Returns:
        SAS token string
    """
    CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
    CONTAINER_NAME = os.getenv("SERVER_CONTAINER_NAME")
    # SERVER_CONTAINER_NAME = os.getenv("GLOBAL_CONTAINER_NAME")
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    # Get account key from connection string
    account_name = blob_service_client.account_name
    account_key = dict(item.split('=', 1) for item in 
                        CONNECTION_STRING.split(';'))['AccountKey']
    
    # Generate SAS token
    sas_token = generate_container_sas(
        account_name=account_name,
        container_name=CONTAINER_NAME,
        account_key=account_key,
        permission=ContainerSasPermissions(
            read=True,
            write=True,
            list=True,
        ),
        expiry=datetime.utcnow() + timedelta(days=duration_days)
    )
    
    return sas_token, account_name, CONTAINER_NAME


# Example usage
def main():
    # Generate SAS token
    sas_token, account_name, container_name = generate_sas_token()
    print(f"Generated SAS Token: {sas_token}")
    
    # Full URL with SAS token for container access
    CLIENT_ACCOUNT_URL = f"https://{account_name}.blob.core.windows.net?{sas_token}"
    print(f"Container URL with SAS: {CLIENT_ACCOUNT_URL}")

if __name__ == "__main__":
    main()