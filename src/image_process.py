import json
from typing import Dict, Optional
import matplotlib.pyplot as plt
import torch
from torchvision.io import decode_image
import utils

def show_image(
        tensor: torch.Tensor, 
        bboxes: Optional[Dict[int, torch.Tensor]] = None,
        pair_id: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        ) -> None:
    """
    Display a single image tensor.

    Args:
        tensor (torch.Tensor): A 3D tensor representing an image (C, H, W).
    """
    if tensor.dim() != 3:
        raise ValueError("Input tensor must be a 3D tensor representing an image (C, H, W).")
    
    # Convert tensor to numpy array and transpose to (H, W, C)
    image = tensor.permute(1, 2, 0).cpu().numpy()
    cmap = plt.get_cmap('viridis')
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')

    if bboxes is not None:
        for style_id, bbox_tensor in bboxes.items():
            if bbox_tensor.size(0) != 4:
                raise ValueError("1D Bounding box tensor must be of size 4.")
            
            color = cmap(style_id*0.5)
            x, y, w, h = bbox_tensor
            rect = plt.Rectangle((x, y), w, h, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            ax.text(x*1.05, y*1.05, f"Style {style_id}", color=color)
    
    if pair_id is not None:
        title = f"Pair ID: {pair_id}."
        ax.set_title(title, fontsize=10)


def load_image_and_metadata(image_id: str, image_dir: str, metadata_dir: str):
    """
    Load an image and its associated bounding box and metadata.

    Args:
        image_id (str): Identifier for the image to be loaded.
        image_dir (str): Directory where images are stored.
        metadata_dir (str): Directory where metadata files are stored.
    Returns:
        image (torch.Tensor): Loaded image tensor.
        bbox (Dict[int: torch.Tensor]): Associated bounding box with style id as key.
        pair_id (int)
    """
    image_path = f"{image_dir}/{image_id}.jpg"
    metadata_path = f"{metadata_dir}/{image_id}.json"

    # Load image
    image_tensor = decode_image(image_path)

    # Load metadata
    with open(metadata_path, 'r') as f:
        img_metadata = json.load(f)

    bboxes = {}
    for k in img_metadata.keys():
        if 'item' in k:
            bbox_coords = img_metadata[k]['bounding_box']
            bbox_coords = torch.tensor(bbox_coords)
            bbox_tensor = utils.post_process_bboxes(bbox_coords)

            bbox_style = img_metadata[k]['style']

            bboxes[bbox_style] = bbox_tensor

    pair_id = img_metadata['pair_id']

    return image_tensor, bboxes, pair_id

def segment_per_bbox(image_tensor: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    '''
    Segment the image using a given bounding box.
    Args:
        image_tensor (torch.Tensor): A 3D tensor representing an image (C, H, W).
        bbox (torch.Tensor): A 1D tensor representing a bounding box (x, y, w, h).
    Returns:
        segmented_image (torch.Tensor): The segmented image tensor.
    '''
    x, y, w, h = bbox
    segmented_image = image_tensor[:, y:y+h, x:x+w]

    return segmented_image
