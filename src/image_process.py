import matplotlib.pyplot as plt
import torch

def show_image(tensor: torch.Tensor) -> None:
    """
    Display a single image tensor.

    Args:
        tensor (torch.Tensor): A 3D tensor representing an image (C, H, W).
    """
    if tensor.dim() != 3:
        raise ValueError("Input tensor must be a 3D tensor representing an image (C, H, W).")
    
    # Convert tensor to numpy array and transpose to (H, W, C)
    image = tensor.permute(1, 2, 0).cpu().numpy()

    plt.imshow(image)
    plt.axis('off')
    plt.show()