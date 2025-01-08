"""
This script remaps pixel-level labels in image datasets. It takes an input
directory of labeled images, a YAML file specifying the mapping of old labels
to new labels, and outputs the remapped images to a specified directory.

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
import argparse


def load_yaml(file_path):
    """Load a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def remap_labels(image, mapping):
    """Remap the labels in the image using the mapping dictionary.

    Args:
        image (np.ndarray): Grayscale image with original class labels.
        mapping (dict): Dictionary mapping original class IDs to new class IDs.

    Returns:
        np.ndarray: Image with remapped class labels.
    """
    remapped_image = np.zeros_like(image, dtype=np.uint8)
    for original_class, new_class in mapping.items():
        remapped_image[image == original_class] = new_class
    return remapped_image


def process_images(input_dir, subdirs, output_dir, mapping):
    """Process and remap all images in the input directory and save them to the output directory.

    Args:
        input_dir (str): Root dataset directory.
        subdirs (list): List of subdirectory names to process within the root dataset directory.
        output_dir (str): Root directory to save remapped images.
        mapping (dict): Dictionary mapping original class IDs to new class IDs.
    """
    for subdir in subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.exists(subdir_path):
            continue

        os.makedirs(output_dir, exist_ok=True)

        for root, _, files in tqdm(os.walk(subdir_path), desc=f"Processing directories in {subdir_path}"):
            for file in tqdm(files, desc=f"Processing files in {root}"):
                if file.lower().endswith(".png"):
                    input_image_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, input_dir)
                    output_image_path = os.path.join(output_dir, relative_path, file)
                    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                    # Load the image
                    image = np.array(Image.open(input_image_path))

                    # Remap the labels
                    remapped_image = remap_labels(image, mapping)

                    # Save the remapped image
                    remapped_image = Image.fromarray(remapped_image)
                    remapped_image.save(output_image_path)


def main(args):
    """Main function to execute the remapping process."""
    # Load the mapping
    mapping_data = load_yaml(args.mapping_file)

    # Convert the mapping from names to IDs
    mapping = {}
    new_class_id = 0
    for new_class, old_classes in mapping_data.items():
        for old_class in old_classes:
            mapping[old_class] = new_class_id
        new_class_id += 1

    # Process and remap the images
    process_images(args.input_dir, args.subdirs, args.output_dir, mapping)
    print("Label remapping completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remap labels in semantic segmentation datasets.")
    parser.add_argument("input_dir", type=str, help="Path to the dataset root directory.")
    parser.add_argument("output_dir", type=str, help="Path to the root output directory for remapped images.")
    parser.add_argument("mapping_file", type=str, help="Path to the YAML file specifying the label mapping.")
    parser.add_argument("--subdirs", nargs='+', default=["Rellis_3D_pylon_camera_node_label_id"],
                        help="List of subdirectory names to process within the root dataset directory.")

    args = parser.parse_args()
    main(args)
