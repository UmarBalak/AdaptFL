* `admin_client_registration.py`
run by admin to register clients.

* `data_service.py`
check for raw data, train model and upload weights on the blob.

* `main_server.py`
this is an api: check for new model weights from client in the blob, if found, aggregate them and save new weights to the blob again (to different container), then notify all the clients about the new global weights.

* `websocket_service.py`
continuously listens to the server, if notification received, download global weights from the server.

## Point to be noted
* server filter new weights based on timestamp (present in the client's weight file name).
* each time client connect to the server, server shares the latest global file version, then client compare latest version with its latest downloaded file, 
* if server's file is latest, client download the new weights from the blob.
* if client has latest file, then it continue to listen for another notification from the server about the new global weights.

## Flow
### Admin Registers a New Client (CSN)
1. Admin generates a unique CSN for the client and sends a registration request to the server.
2. Server generates client_id and api_key and both server and admin stores them securely.
---

### Admin Configures the Client Machine
Admin sets up the client machine with:
* CSN
* client_id
* api_key
* Blob credentials
---

### Client Authentication and Usage
1. Client directly uses the pre-configured client_id, api_key, and blob credentials for operations.
