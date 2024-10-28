"""
Script to extract images from a specified topic of type "Image" in a ROS2 bag file and save them in an output directory.

This script uses the `argparse` library to take command-line arguments for input and output paths, the topic name, 
and optional parameters for time filtering and frequency of image extraction. Images are saved in PNG format, 
with their filenames based on their timestamps.

Usage:
    python extract_images.py <bag_path> <images_topic> <output_dir> [--time_start TIME_START] [--time_end TIME_END] [--every_k EVERY_K]

Example:
    python extract_images.py /path/to/bag /topic/to/extract /output/dir --time_start 0 --time_end -1 --every_k 10

Dependencies:
    - rosbags: for reading ROS2 bag files.
    - rclpy: for message deserialization.
    - sensor_msgs: for ROS2 image message definitions.
    - cv_bridge: for converting ROS images to OpenCV format.
    - OpenCV: for image processing and saving.
"""

from rosbags.rosbag2 import Reader
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import argparse

def extract_images_from_bag(bag_path: str, images_topic: str, output_dir: str, time_start: float = 0.0, time_end: float = -1.0, every_k: int = 1) -> None:
    """
    Extracts images from a specified topic of a ROS2 bag file and saves them in an output directory.

    Args:
        bag_path (str): Path to the ROS2 bag file from which images are to be extracted.
        images_topic (str): Topic name where the images are stored.
        output_dir (str): Directory path where the extracted images will be saved.
        time_start (float): Earliest timestamp for images to be considered (default: 0.0).
        time_end (float): Latest timestamp for images to be considered (-1.0 means no end limit).
        every_k (int): Frequency for saving images, i.e., every k-th image (default: 1).
    """
    i = 0
    with Reader(bag_path) as reader:
        for connection, _, rawdata in reader.messages():
            if connection.topic == images_topic:
                msg = deserialize_message(rawdata, Image)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 10 ** 9

                # Disregard images outside the specified time range
                if time_end == -1.0:
                    if time_start > timestamp:
                        continue
                else:
                    if not (time_start <= timestamp <= time_end):
                        continue
                
                i += 1
                if i % every_k != 1:
                    continue

                bgr_img = CvBridge().imgmsg_to_cv2(img_msg=msg, desired_encoding="bgr8")
                bgr_img = cv2.rotate(bgr_img, cv2.ROTATE_90_CLOCKWISE)

                # Create output directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                file_name = f"{msg.header.stamp.sec}_{msg.header.stamp.nanosec}.png"
                output_path = os.path.join(output_dir, file_name)
                cv2.imwrite(output_path, bgr_img)
                print(f"Saved image: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from a ROS2 bag file topic and save them to a specified directory.")
    parser.add_argument("bag_path", type=str, help="Path to the ROS2 bag file.")
    parser.add_argument("images_topic", type=str, help="Name of the topic containing the images.")
    parser.add_argument("output_dir", type=str, help="Output directory to save extracted images.")
    parser.add_argument("--time_start", type=float, default=0.0, help="Start time for extraction (default: 0.0).")
    parser.add_argument("--time_end", type=float, default=-1.0, help="End time for extraction (-1.0 means no limit).")
    parser.add_argument("--every_k", type=int, default=1, help="Frequency to save images, e.g., every k-th image (default: 1).")

    args = parser.parse_args
