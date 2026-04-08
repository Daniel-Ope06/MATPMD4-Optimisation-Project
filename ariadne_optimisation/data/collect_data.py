"""
Note: The core computer vision and pathfinding modules
(TargetDetector, UNetSegmenter, PathFinder)
are imported from the Ariadne Proof of Concept  repository.

Source: https://github.com/Daniel-Ope06/ariadne-poc
"""

import sys
import os
import glob
import time
import csv
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from ariadne_optimisation.modules.target_detector import TargetDetector  # noqa: E402, E501, I001
from ariadne_optimisation.modules.unet_segmenter import UNetSegmenter  # noqa: E402, E501, I001
from ariadne_optimisation.modules.path_finder import PathFinder  # noqa: E402, E501, I001

YOLO_WEIGHTS = os.path.join(
    ROOT_DIR, "ariadne_optimisation", "models",
    "maze_actor_detector_yolo26s_v1.pt"
)

UNET_WEIGHTS = os.path.join(
    ROOT_DIR, "ariadne_optimisation", "models",
    "maze_floor_segmenter_unet_v1.pth"
)

MAZES_DIR = os.path.join(
    ROOT_DIR, "ariadne_optimisation", "mazes"
)


def is_path_crossing_wall(optimal_path, high_res_wall_mask, grid_resolution=640, img_size=640):  # noqa: E501
    """
    Checks if the computed path intersects with any walls.
    """
    # Create a blank image to draw the path on (0 = Black)
    path_mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # Calculate how many pixels each R x R grid cell represents
    scale = img_size / grid_resolution

    # Draw the path as a 1-pixel thick line connecting the nodes
    for i in range(len(optimal_path) - 1):
        # Shift by scale/2 to put the point
        # exactly in the center of the grid cell
        p1 = (int(optimal_path[i][0] * scale + scale/2), int(optimal_path[i][1] * scale + scale/2))  # noqa: E501
        p2 = (int(optimal_path[i+1][0] * scale + scale/2), int(optimal_path[i+1][1] * scale + scale/2))  # noqa: E501

        # Draw path in pure white (255)
        cv2.line(path_mask, p1, p2, color=255, thickness=1)

    # Bitwise AND between the drawn path (255) and the physical walls (255)
    # If the path touches a wall, those pixels will remain 255.
    # If it doesn't touch a wall, those pixels will become 0.
    collision = cv2.bitwise_and(path_mask, high_res_wall_mask)

    if np.sum(collision) > 0:
        return True  # Wall crossed
    return False


def main():
    mazes = glob.glob(os.path.join(MAZES_DIR, "*.png"))
    total_mazes = len(mazes)

    csv_file_path = os.path.join(SCRIPT_DIR, "data.csv")
    log_file_path = os.path.join(SCRIPT_DIR, "failed_mazes.txt")

    # --- Pre-compute High-Res Wall Masks ---
    print("[*] Pre-computing 640x640 Wall Masks for collision detection...")
    segmenter_640 = UNetSegmenter(UNET_WEIGHTS, grid_size=640)
    high_res_walls = {}

    for img_path in mazes:
        filename = os.path.basename(img_path)
        # Segmenter returns 1=Path, 0=Wall
        raw_mask_640 = segmenter_640.generate_matrix(img_path)
        # Convert to a collision mask where Wall=255, Path=0
        wall_mask = (raw_mask_640 == 0).astype(np.uint8) * 255
        high_res_walls[filename] = wall_mask

    print("[+] Pre-computation complete.\n")

    # Define search space: R from 10 to 100.
    resolutions_to_test = range(10, 101, 2)

    results = []

    # Open log file to write failures as they happen
    with open(log_file_path, "w") as log_file:
        log_file.write("--- Path Occlusion / Failure Log ---\n\n")

        for R in resolutions_to_test:
            print(f"[*] Testing Resolution: {R}x{R}...")

            # Re-initialize with the new grid size
            detector = TargetDetector(YOLO_WEIGHTS, grid_size=R)
            segmenter = UNetSegmenter(UNET_WEIGHTS, grid_size=R)
            path_finder = PathFinder()

            failures = 0
            cumulative_time = 0.0
            failed_mazes = []

            for img_path in mazes:
                filename = os.path.basename(img_path)

                # Detect Targets
                targets = detector(img_path)
                ugv_data = targets.get(0)
                human_data = targets.get(1)

                start_node = ugv_data['grid_node']  # type: ignore
                goal_node = human_data['grid_node']  # type: ignore

                # Segment Image
                binary_matrix = segmenter.generate_matrix(img_path)

                # Measure ONLY the pathfinding execution time
                start_time = time.perf_counter()
                optimal_path = path_finder.find_path(
                    binary_matrix, start_node, goal_node
                )
                end_time = time.perf_counter()

                cumulative_time += (end_time - start_time)

                # Check for Failure (Occlusion)
                if not optimal_path:
                    # Failure 1: Couldn't find a route at all
                    failures += 1
                    failed_mazes.append(f"{filename} (No Path)")

                elif is_path_crossing_wall(optimal_path, high_res_walls[filename], R):  # noqa: E501
                    # Failure 2: Found a route, but it crossed a physical wall
                    failures += 1
                    failed_mazes.append(f"{filename} (Wall Crossing)")

            # Calculate metrics for this resolution
            avg_time_ms = (cumulative_time / total_mazes) * \
                1000  # Convert to milliseconds
            occlusion_rate = failures / total_mazes

            # Store and print results
            results.append([R, avg_time_ms, occlusion_rate])
            print(f"    -> Avg Time: {avg_time_ms:.2f} ms | Occlusion Rate: {occlusion_rate:.2f}")  # noqa: E501

            # Log the specific failures
            # for manual verification of wall-crossings
            if failures > 0:
                log_file.write(f"Resolution R={R} ({failures} failures):\n")
                for maze in failed_mazes:
                    log_file.write(f"  - {maze}\n")
                log_file.write("\n")

    # Export the numerical data to a CSV for plotting
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Resolution (R)", "Average Compute Time (ms)", "Occlusion Rate"])
        writer.writerows(results)

    print("\n[+] Data collection complete.")
    print(f"    -> CSV saved to: {csv_file_path}")
    print(f"    -> Failure log saved to: {log_file_path}")


if __name__ == "__main__":
    main()
