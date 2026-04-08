"""
Wall Crossing Visual Verification Script

This script parses the optimization failure log, isolates mazes that
triggered a Wall Crossing failure, and visually renders
the computed path at the specific failure resolution for manual verification.
"""

import sys
import os
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from ariadne_optimisation.modules.target_detector import TargetDetector  # noqa: E402, E501, I001
from ariadne_optimisation.modules.unet_segmenter import UNetSegmenter  # noqa: E402, E501, I001
from ariadne_optimisation.modules.path_finder import PathFinder  # noqa: E402, E501, I001
from ariadne_optimisation.modules.visualizer import Visualizer  # noqa: E402, E501, I001

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

LOG_FILE_PATH = os.path.join(
    ROOT_DIR, "ariadne_optimisation", "data",
    "failed_mazes.txt"
)


def parse_log_file(log_path):
    """
    Parses the log file to extract wall crossing failures.
    Returns a dictionary grouped by resolution:
    {R: [filename1, filename2, ...]}
    """
    crossings_by_resolution = {}
    current_R = None

    with open(log_path, "r") as file:
        for line in file:
            line = line.strip()

            # Detect the resolution header,
            # e.g., "Resolution R=36 (2 failures):"
            if line.startswith("Resolution R="):
                # Extract the integer R
                current_R = int(line.split("R=")[1].split()[0])
                if current_R not in crossings_by_resolution:
                    crossings_by_resolution[current_R] = []

            # Detect the specific wall crossing tag
            elif "(Wall Crossing)" in line and current_R is not None:
                # Line format: "- maze_042.png (Wall Crossing)"
                filename = line.replace("-", "").replace("(Wall Crossing)", "").strip()  # noqa: E501
                crossings_by_resolution[current_R].append(filename)

    # Remove any resolutions that had 0 wall crossings
    return {k: v for k, v in crossings_by_resolution.items() if len(v) > 0}  # noqa: E501


def main():
    print("[*] Parsing log file for Wall Crossings...")
    tasks = parse_log_file(LOG_FILE_PATH)

    total_to_verify = sum(len(files) for files in tasks.values())
    print(f"[*] Found {total_to_verify} total wall-crossing instances to visualize.\n")  # noqa: E501

    if total_to_verify == 0:
        print("[+] No wall crossings found in the log. Verification complete.")
        return

    # Process grouped by resolution to minimize model reloading
    for R, filenames in tasks.items():
        print(f"--- Loading Models for Resolution: {R}x{R} ---")
        detector = TargetDetector(YOLO_WEIGHTS, grid_size=R)
        segmenter = UNetSegmenter(UNET_WEIGHTS, grid_size=R)
        path_finder = PathFinder()
        visualizer = Visualizer(grid_size=R)

        for filename in filenames:
            img_path = os.path.join(MAZES_DIR, filename)

            if not os.path.exists(img_path):
                print(f"  [!] Missing image file: {filename}")
                continue

            print(f"  -> Visualizing: {filename} at R={R}")

            # Run Pipeline
            targets = detector(img_path)
            ugv_data = targets.get(0)
            human_data = targets.get(1)

            start_node = ugv_data['grid_node']  # type: ignore
            goal_node = human_data['grid_node']  # type: ignore

            binary_matrix = segmenter.generate_matrix(img_path)
            optimal_path = path_finder.find_path(
                binary_matrix, start_node, goal_node)

            # Generate and save visual output
            final_output = visualizer(
                img_path, targets, optimal_path)  # type: ignore

            # Save format: R30_maze_042.png
            save_name = f"R{R}_{filename}"
            save_path = os.path.join(SCRIPT_DIR, save_name)
            cv2.imwrite(save_path, final_output)  # type: ignore

    print("\n[+] Visual verification generation complete.")
    print(f"    Images saved to: {SCRIPT_DIR}")


if __name__ == "__main__":
    main()
