import cv2
import numpy as np
import os


class Visualizer:
    """
    Overlays bounding boxes and the routing path
    onto the original maze image using OpenCV.
    """

    def __init__(self, img_size=640, grid_size=30):
        self.img_size = img_size
        self.grid_size = grid_size
        self.cell_size = img_size / grid_size

        # Define tactical UI colors in BGR format (OpenCV standard)
        self.colors = {
            "ugv": (255, 255, 0),    # Cyan
            "human": (255, 0, 255),  # Magenta
            "path": (0, 255, 0),     # Lime Green
            "text": (255, 255, 255)  # White
        }

    def __call__(self, image_path, targets, optimal_path):
        """
        Draws the bounding boxes, labels, and path on the image.

        Args:
            image_path (str): Path to the original .png image.
            targets (dict): The dictionary from TargetDetector.
            optimal_path (list): The list of (x, y) tuples from PathFinder.

        Returns:
            numpy array: The final composited image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image missing: {image_path}")

        # Load the original image
        img = cv2.imread(image_path)

        # Draw the A* Routing Path First
        if optimal_path:
            pixel_path = []
            for node in optimal_path:
                # Convert the matrix grid node back to the exact pixel center
                px_x = int((node[0] * self.cell_size) + (self.cell_size / 2))
                px_y = int((node[1] * self.cell_size) + (self.cell_size / 2))
                pixel_path.append([px_x, px_y])

            # Convert list to the specific numpy array format OpenCV requires
            pts = np.array(pixel_path, np.int32).reshape((-1, 1, 2))

            # Draw a thick, anti-aliased polyline
            cv2.polylines(
                img, [pts],  # type: ignore
                isClosed=False, color=self.colors["path"],
                thickness=4, lineType=cv2.LINE_AA)

        # Draw Bounding Boxes and Labels
        for cls_id, data in targets.items():
            top_left_x, top_left_y = data['bbox_top_left']
            width, height = data['bbox_size']

            # Calculate bottom right corner
            bottom_right_x = top_left_x + width
            bottom_right_y = top_left_y + height

            # Determine label and color
            label = "UGV" if cls_id == 0 else "Human"
            box_color = self.colors["ugv"] if cls_id == 0 else self.colors[
                "human"]

            # Draw the bounding box
            cv2.rectangle(
                img,  # type: ignore
                (top_left_x, top_left_y),
                (bottom_right_x, bottom_right_y),
                box_color, 3)

            # Draw a solid background rectangle for the text label
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                img,  # type: ignore
                (top_left_x, top_left_y - text_h - 10),
                (top_left_x + text_w, top_left_y),
                box_color, -1)

            # Add the text label
            cv2.putText(
                img, label,  # type: ignore
                (top_left_x, top_left_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                self.colors["text"], 2, cv2.LINE_AA)

        return img
