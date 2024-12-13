"""
This script remaps pixel-level labels in image datasets. It takes an input directory of labeled images, a YAML file specifying the mapping of old labels to new labels, and outputs the remapped images to a specified directory.

The script:
- Loads the label mapping from a YAML file.
- Iterates through the dataset directory structure, processing each labeled image.
- Applies the mapping to create new label images with updated class IDs.
- Maintains the original directory structure in the output directory.

The YAML file contains one line for each new class.
Order matters: the first new class will be assigned the id 0, and so on.
Line format: "new_class": [old_class_ids]

Example:
water: [6, 31]  # Combine water (6) and puddle (31) into one class
obstacle: [15, 34]  # Combine log (15) and rubble (34) into one class
...
"""
import os
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_yaml(file_path):
    """Load a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def remap_labels(image, mapping):
    """Remap the labels in the image using the mapping dictionary."""
    remapped_image = np.zeros_like(image, dtype=np.uint8)
    for original_class, new_class in mapping.items():
        remapped_image[image == original_class] = new_class
    return remapped_image


def process_images(input_dir, output_dir, mapping):
    """Process and remap all images in the input directory and save them to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for experiment in tqdm(os.listdir(input_dir), desc="Processing experiments"):
        experiment_dir = os.path.join(input_dir, experiment, "pylon_camera_node_label_id")
        if not os.path.isdir(experiment_dir):
            continue

        output_experiment_dir = os.path.join(output_dir, experiment, "pylon_camera_node_label_id")
        os.makedirs(output_experiment_dir, exist_ok=True)

        for filename in os.listdir(experiment_dir):
            if filename.endswith(".png"):
                input_image_path = os.path.join(experiment_dir, filename)
                output_image_path = os.path.join(output_experiment_dir, filename)

                # Load the image
                image = np.array(Image.open(input_image_path))

                # Remap the labels
                remapped_image = remap_labels(image, mapping)

                # Save the remapped image
                remapped_image = Image.fromarray(remapped_image)
                remapped_image.save(output_image_path)


if __name__ == "__main__":
    # Define paths
    input_directory = "/home/rbertoglio/datasets/Rellis_3D/Rellis_3D_pylon_camera_node_label_id/Rellis-3D"
    output_directory = "/home/rbertoglio/datasets/Rellis_3D_Remapped"
    mapping_file = "/home/rbertoglio/datasets/Rellis_3D/mapping.yaml"  # Path to the YAML file containing the mapping

    # Load the mapping
    mapping_data = load_yaml(mapping_file)

    # Convert the mapping from names to IDs (assumes mapping is like {"new_class": [old_class_ids]})
    mapping = {}
    new_class_id = 0
    for new_class, old_classes in mapping_data.items():
        for old_class in old_classes:
            mapping[old_class] = new_class_id
        new_class_id += 1

    # Process and remap the images
    process_images(input_directory, output_directory, mapping)

    print("Label remapping completed!")
