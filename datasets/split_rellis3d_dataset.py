import os
import shutil

# Define paths
dataset_dir = "/home/rbertoglio/datasets/Rellis_3D"
split_dir = os.path.join(dataset_dir, "Rellis_3D_image_split")
output_dirs = {
    "train": os.path.join(dataset_dir, "Rellis_3D_train"),
    "test": os.path.join(dataset_dir, "Rellis_3D_test"),
    "val": os.path.join(dataset_dir, "Rellis_3D_val"),
}

# Ensure output directories exist
for split, out_dir in output_dirs.items():
    os.makedirs(os.path.join(out_dir, "Rellis_3D_pylon_camera_node"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "Rellis_3D_pylon_camera_node_label_id"), exist_ok=True)


# Function to process each split
def process_split(split_name, lst_file):
    """
    Process a given split and copy images and labels to the respective directories.

    Parameters:
        split_name (str): Name of the split (e.g., 'train', 'test', 'val').
        lst_file (str): Path to the .lst file containing the split information.
    """
    with open(lst_file, "r") as file:
        lines = file.readlines()

    for line in lines:
        image_path, label_path = line.strip().split()

        # Define source paths
        src_image = os.path.join(dataset_dir, "Rellis_3D_pylon_camera_node", "Rellis-3D", image_path)
        src_label = os.path.join(dataset_dir, "Rellis_3D_pylon_camera_node_label_id", "Rellis-3D", label_path)

        # Define destination base paths
        dest_image_base = os.path.join(output_dirs[split_name], "Rellis_3D_pylon_camera_node", "Rellis-3D")
        dest_label_base = os.path.join(output_dirs[split_name], "Rellis_3D_pylon_camera_node_label_id", "Rellis-3D")

        # Extract the relative paths for image and label
        rel_image_path = os.path.dirname(image_path)  # Relative directory of the image
        rel_label_path = os.path.dirname(label_path)  # Relative directory of the label

        # Define the full destination paths, preserving directory structure
        dest_image = os.path.join(dest_image_base, rel_image_path, os.path.basename(image_path))
        dest_label = os.path.join(dest_label_base, rel_label_path, os.path.basename(label_path))

        # Create the necessary directories for the destination paths
        os.makedirs(os.path.dirname(dest_image), exist_ok=True)
        os.makedirs(os.path.dirname(dest_label), exist_ok=True)

        # Copy files to the respective directories
        if os.path.exists(src_image) and os.path.exists(src_label):
            shutil.copy2(src_image, dest_image)
            shutil.copy2(src_label, dest_label)
        else:
            print(f"Warning: File {src_image} or {src_label} does not exist and will be skipped.")


# Process all splits
for split in ["train", "test", "val"]:
    lst_file = os.path.join(split_dir, f"{split}.lst")
    if os.path.exists(lst_file):
        process_split(split, lst_file)
    else:
        print(f"Error: Split file {lst_file} not found.")

print("Dataset splitting completed.")
