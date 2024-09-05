import os

def create_project_structure(root_dir="decifer"):
    # Define the folder structure
    folder_structure = {
        "data/": {
            "chili100k/": {
                "raw/": {},
                "preprocessed/": {},
                "xrd_data/": {},
                "embeddings/": {}
            },
        },
        "models/": {},
        "experiments/": {
            "unconditioned/": {
                "experiment_01/": {}
            },
            "prefix_conditioning/": {
                "experiment_01/": {}
            },
            "encoded_conditioning/": {
                "experiment_01/": {}
            }
        },
        "scripts/": {},
        "checkpoints/": {},
        "docs/": {}
    }

    def create_folders(base_path, structure):
        for folder, subfolders in structure.items():
            current_path = os.path.join(base_path, folder)
            if not os.path.exists(current_path):
                os.makedirs(current_path)
                print(f"Created: {current_path}")
            else:
                print(f"Already exists: {current_path}")
            create_folders(current_path, subfolders)

    # Create the folder structure if not already there
    create_folders(root_dir, folder_structure)

# Run the script to create the structure
create_project_structure()
