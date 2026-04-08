import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp  # type: ignore


class UNetSegmenter:
    """
    Semantic segmentation using U-Net.
    Outputs a binary navigable matrix for A* pathfinding.
    """

    def __init__(self, weights_path, grid_size=30, img_size=(640, 640)):
        self.img_size = img_size
        self.grid_size = grid_size
        self.cell_size = img_size[0] // grid_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Instantiate the architecture
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )

        # Load the trained parameters and lock the model into evaluation mode
        self.model.load_state_dict(torch.load(
            weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def generate_matrix(self, img_path):
        """
        Executes a forward pass to generate the binary floor mask.
        Returns the matrix (1=Path, 0=Wall).
        """
        # Load and preprocess (match the Colab Dataset class)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
        image = cv2.resize(image, self.img_size)

        # Normalize and convert to PyTorch Tensor format
        image_tensor = image.astype(np.float32) / 255.0
        image_tensor = np.transpose(image_tensor, (2, 0, 1))

        input_tensor = torch.tensor(image_tensor).unsqueeze(0).to(self.device)

        # Neural Inference
        with torch.no_grad():
            raw_pred = self.model(input_tensor)
            prob_mask = torch.sigmoid(raw_pred)

            # 640x640 2D matrix
            pixel_mask = (prob_mask > 0.5).float().cpu().squeeze().numpy()

        pixel_mask_uint8 = pixel_mask.astype(np.uint8)

        # Enforce the OpenCV Odd Number Rule (kernels need a true center pixel)
        if self.cell_size % 2 == 0:
            kernel_size = self.cell_size + 1
        else:
            kernel_size = self.cell_size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Erode the floor (1s) to artificially thicken the walls (0s)
        buffered_mask = cv2.erode(pixel_mask_uint8, kernel, iterations=1)

        # Compress down to the grid size (30x30)
        binary_mask = cv2.resize(
            buffered_mask,
            (self.grid_size, self.grid_size),
            interpolation=cv2.INTER_NEAREST
        )

        return binary_mask
