"""
This script generates grayscale label images for semantic segmentation tasks.

Each grayscale label image has pixel values corresponding to class IDs,
which are defined in a colormap text file.
The script reads RGB annotation images and maps their pixel values to
grayscale class IDs using the colormap. The output images retain the directory
structure of the input annotation images.

The colormap text file should have the following format:
    <class_id> <class_name> <R> <G> <B>

Usage:
    python rgb_to_grayscale_labels.py <input_dir> <colormap_file> <output_dir>

Arguments:
    input_dir: Path to the directory containing RGB annotation images,
               organized in subdirectories.
    colormap_file: Path to the text file defining class IDs and their
                   corresponding RGB values.
    output_dir: Path to the directory where grayscale label images will be saved.
"""
import os

import cv2
import numpy as np
from tqdm import tqdm


def load_colormap(colormap_file):
    """
    Loads the class ID to RGB mapping from a colormap text file.

    Args:
        colormap_file (str): Path to the colormap text file.

    Returns:
        dict: A dictionary mapping RGB tuples to class IDs.
    """
    colormap = {}
    with open(colormap_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            rgb = tuple(map(int, parts[2:]))
            colormap[rgb] = class_id
    return colormap


def process_image(input_path, colormap):
    """
    Converts an RGB annotation image to a grayscale label image using
    a colormap.

    Args:
        input_path (str): Path to the RGB annotation image.
        colormap (dict): A dictionary mapping RGB tuples to class IDs.

    Returns:
        np.ndarray: A 2D array representing the grayscale label image.
    """
    rgb_image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    height, width, _ = rgb_image.shape
    grayscale_image = np.zeros((height, width), dtype=np.uint8)

    for rgb, class_id in colormap.items():
        mask = (rgb_image[:, :, 0] == rgb[2]) & (rgb_image[:, :, 1] == rgb[1]) & (rgb_image[:, :, 2] == rgb[0])
        grayscale_image[mask] = class_id

    return grayscale_image


def process_directory(input_dir, colormap, output_dir):
    """
    Processes all RGB annotation images in a directory structure and generates
    grayscale label images.

    Args:
        input_dir (str): Path to the root input directory containing RGB
                         annotation images.
        colormap (dict): A dictionary mapping RGB tuples to class IDs.
        output_dir (str): Path to the root output directory to save grayscale
                          label images.

    Returns:
        None
    """
    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):
                files_to_process.append(os.path.join(root, file))

    for input_path in tqdm(files_to_process, desc="Processing images"):
        relative_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, relative_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        grayscale_image = process_image(input_path, colormap)
        cv2.imwrite(output_path, grayscale_image)


def main(input_dir, colormap_file, output_dir):
    """
    Main function to generate grayscale label images from RGB annotation images.

    Args:
        input_dir (str): Path to the root directory containing RGB annotation images.
        colormap_file (str): Path to the text file defining the colormap.
        output_dir (str): Path to the root directory for saving grayscale images.

    Returns:
        None
    """
    colormap = load_colormap(colormap_file)
    process_directory(input_dir, colormap, output_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate grayscale label images for semantic segmentation.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing RGB annotation images.")
    parser.add_argument("colormap_file", type=str, help="Path to the colormap text file.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for grayscale label images.")

    args = parser.parse_args()

    main(args.input_dir, args.colormap_file, args.output_dir)
