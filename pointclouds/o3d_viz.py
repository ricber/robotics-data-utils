import argparse
import open3d as o3d
import numpy as np
import csv
import os


def load_csv_points(csv_file, fixed_z):
    """
    Load points from a CSV file.
    Assumes the CSV has a header with columns including 'E' and 'N'
    for the x and y coordinates. The z coordinate is set to the provided fixed value.
    The CSV is assumed to be tab-delimited.
    """
    csv_points = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                x = float(row['E'])
                y = float(row['N'])
                z = fixed_z  # use the fixed z value instead of the U column
                csv_points.append([x, y, z])
            except Exception as e:
                print("Skipping row {} due to error: {}".format(row, e))
    if csv_points:
        csv_points_np = np.array(csv_points)
        csv_pcd = o3d.geometry.PointCloud()
        csv_pcd.points = o3d.utility.Vector3dVector(csv_points_np)
        # Paint CSV points in red for differentiation.
        csv_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        return csv_pcd
    else:
        print("No valid points found in CSV.")
        return None


def visualize_with_visualizer(ply_file, point_size, csv_file, fixed_z, show_frame, frame_size):
    """
    Visualize point clouds using the Open3D Visualizer.
    Optionally loads a PLY file and a CSV file (with fixed z) and displays them together.
    If show_frame is True, a coordinate reference frame is added.
    """
    geometries = []

    if ply_file is not None:
        if os.path.exists(ply_file):
            pcd_ply = o3d.io.read_point_cloud(ply_file)
            geometries.append(pcd_ply)
        else:
            print("PLY file {} does not exist.".format(ply_file))

    if csv_file is not None:
        if os.path.exists(csv_file):
            csv_pcd = load_csv_points(csv_file, fixed_z)
            if csv_pcd is not None:
                geometries.append(csv_pcd)
        else:
            print("CSV file {} does not exist.".format(csv_file))

    if show_frame:
        # Create a coordinate frame.
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=[0, 0, 0])
        geometries.append(coord_frame)

    if not geometries:
        print("No geometry loaded. Exiting.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geom in geometries:
        vis.add_geometry(geom)

    # Set render options (e.g., point size)
    render_option = vis.get_render_option()
    render_option.point_size = point_size

    vis.run()
    vis.destroy_window()


def visualize_with_editing(ply_file, output_dir):
    """
    Visualize a PLY file using Open3D's editing tool, allowing interactive point picking.
    After picking, the indices and coordinates of the selected points are printed and
    saved in .ply and .csv files.
    """
    # Load the point cloud.
    pcd = o3d.io.read_point_cloud(ply_file)

    # Print instructions for point picking.
    print("Instructions for point picking:")
    print("  - Hold SHIFT and left click to select points.")
    print("  - Hold SHIFT and right click to undo point picking.")
    print("  - Press 'q' or 'Esc' to finish and exit.")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    picked_indices = vis.get_picked_points()

    # Extract coordinates of picked points
    picked_points = np.asarray(pcd.points)[picked_indices]

    if len(picked_points) == 0:
        print("[WARNING] No points selected.")
    else:
        # Create output folder if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save to .ply
        picked_pcd = o3d.geometry.PointCloud()
        picked_pcd.points = o3d.utility.Vector3dVector(picked_points)
        ply_path = os.path.join(output_dir, "picked_points.ply")
        o3d.io.write_point_cloud(ply_path, picked_pcd)
        print(f"[INFO] Saved picked points to: {ply_path}")

        # Save to .csv
        csv_path = os.path.join(output_dir, "picked_points.csv")
        np.savetxt(csv_path, picked_points, delimiter=",", header="x,y,z", comments='')
        print(f"[INFO] Saved coordinates to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description=("Visualize point clouds from a PLY file and/or a CSV file using Open3D. "
                     "The CSV file should contain columns 'E' and 'N' for x and y coordinates. "
                     "The z coordinate is fixed to the value provided with --fixed_z.")
    )
    parser.add_argument("--ply", type=str, help="Path to the PLY file.", default=None)
    parser.add_argument("--csv", type=str, help="Path to the CSV file.", default=None)
    parser.add_argument(
        "--mode",
        choices=["visualizer", "editing"],
        default="visualizer",
        help=("Visualization mode: 'visualizer' uses the Visualizer class (with customizable point size); "
              "'editing' allows interactive point picking (only applies to the first loaded geometry).")
    )
    parser.add_argument("-o", "--output_dir", default="output", help="Directory to save the picked points in 'editing' mode")
    parser.add_argument(
        "--point_size",
        type=float,
        default=2.0,
        help="Point size to use in 'visualizer' mode (default: 2.0)."
    )
    parser.add_argument(
        "--fixed_z",
        type=float,
        default=0.0,
        help="Fixed z value to assign to all CSV points (default: 0.0)."
    )
    parser.add_argument(
        "--show_frame",
        action="store_true",
        help="If set, adds a coordinate reference frame to the visualization."
    )
    parser.add_argument(
        "--frame_size",
        type=float,
        default=5.0,
        help="Size of the coordinate reference frame (default: 1.0)."
    )
    args = parser.parse_args()

    if args.ply is None and args.csv is None:
        parser.error("At least one of --ply or --csv must be provided.")

    if args.mode == "visualizer":
        visualize_with_visualizer(args.ply, args.point_size, args.csv, args.fixed_z, args.show_frame, args.frame_size)
    elif args.mode == "editing":
        visualize_with_editing(args.ply, args.output_dir)


if __name__ == "__main__":
    main()
