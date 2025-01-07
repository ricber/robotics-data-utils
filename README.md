# Robotics data utils
Python scripts to extract, manipulate and visualize robotics data.

### Folders structure

```
robotics-data-utils
└─── ROS2
│   | ros2_bag_extract_images.py: Script to extract RGB images from a ROS2 bag
└─── datasets
|   | remap_rellis3d_labels.py: This script remaps pixel-level labels using a specified mapping from an external YAML file, combining or reassigning classes
|   | rgb_to_grayscale_labels.py: This script converts RGB semantic segmentation annotations to grayscale label images, where pixel values correspond to class IDs.
│   | split_rellis3d_dataset.py: This script organizes the Rellis_3D dataset by splitting it into training, testing, and validation subsets based on predefined .lst files
```
