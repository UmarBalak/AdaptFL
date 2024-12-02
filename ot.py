import os

# Define project folder structure
folders = [
    "data",
    "global_server",
    "client",
    "models",
    "utils",
    "logs"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("Project folder structure created.")
