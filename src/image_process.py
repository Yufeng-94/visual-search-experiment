import json
from typing import Dict, Optional
import matplotlib.pyplot as plt
import torch
from torchvision.io import decode_image
import utils

METADATA_ITEM_KEYS = ['scale', 'viewpoint', 'zoom_in', 'style', 'bounding_box', 'occlusion', 'category_id']

def show_image(
        tensor: torch.Tensor, 
        bboxes: Optional[Dict[int, list]] = None,
        pair_id: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        ) -> None:
    """
    Display a single image tensor.
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
        for style_id, bbox_coord in bboxes.items():
            if len(bbox_coord) != 4:
                raise ValueError("1D Bounding box tensor must be of size 4.")
            
            color = cmap(style_id*0.5)
            x, y, w, h = bbox_coord
            rect = plt.Rectangle((x, y), w, h, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            ax.text(x*1.05, y*1.05, f"Style {style_id}", color=color)
    
    if pair_id is not None:
        title = f"Pair ID: {pair_id}."
        ax.set_title(title, fontsize=10)


def load_image_and_metadata(
        image_name: str, 
        image_dir: str, 
        metadata_dir: str,
        return_bbox: bool=False,
        ):
    """
    Load an image and its associated bounding box and metadata.

    Args:
        image_name (str): Identifier for the image to be loaded.
        image_dir (str): Directory where images are stored.
        metadata_dir (str): Directory where metadata files are stored.
        return_bbox (bool): Whether to return bounding box information.
    Returns:
        image (torch.Tensor): Loaded image tensor.
        metadata (dict): Associated metadata.
        bboxes (dict, optional): Bounding box information if return_bbox is True.
    """
    image_path = f"{image_dir}/{image_name}.jpg"
    metadata_path = f"{metadata_dir}/{image_name}.json"

    # Load image
    image_tensor = decode_image(image_path)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata_raw = json.load(f)

    metadata = {}
    for k, v in metadata_raw.items():
        if 'item' in k:
            item_metadata = {}
            for i_k in METADATA_ITEM_KEYS:
                if 'bbox' in i_k: # post process bounding box
                    item_metadata[i_k] = utils.post_process_bboxes(v.get(i_k, []))
                else:
                    item_metadata[i_k] = v.get(i_k, None)
            metadata[k] = item_metadata
        else:
            metadata[k] = v

    # Add other metadata
    ## NOTE: could replace with Path
    metadata['image_name'] = image_name
    metadata['image_path'] = image_path

    if return_bbox:
        bboxes = {}
        for k, v in metadata_raw.items():
            if 'item' in k and 'bounding_box' in v:
                bboxes[v['style']] = utils.post_process_bboxes(v['bounding_box'])

        return image_tensor, metadata, bboxes

    return image_tensor, metadata

def segment_per_bbox(image_tensor: torch.Tensor, bbox: list) -> torch.Tensor:
    '''
    Segment the image using a given bounding box.
    Args:
        image_tensor (torch.Tensor): A 3D tensor representing an image (C, H, W).
        bbox (list): A list representing a bounding box [x, y, w, h].
    Returns:
        segmented_image (torch.Tensor): The segmented image tensor.
    '''
    x, y, w, h = bbox
    segmented_image = image_tensor[:, y:y+h, x:x+w]

    return segmented_image
