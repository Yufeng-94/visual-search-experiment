import os
from typing import Optional
import json

ITEM_KEYS = ['scale', 'viewpoint', 'zoom_in', 'style', 'bounding_box', 'occlusion', 'category_id']

def extract_image_metadata(
        image_name: str, 
        metadata_dir: str, 
        image_dir: str,
        style: Optional[int]=None
        ) -> dict:
    """
    Extract metadata from a JSON file associated with the image.
    """
    # Load metadata JSON
    metadata_path = os.path.join(metadata_dir, f"{image_name}.json")
    with open(metadata_path, 'r') as f:
        metadata_raw = json.load(f)

    # Prepare extracted metadata
    metadata = {}

    if not style: # If style is not specified
        metadata['segmented'] = False

        for k, v in metadata_raw.items():
            if 'item' in k:
                item_metadata = {}
                for i_k in ITEM_KEYS:
                    item_metadata[i_k] = v.get(i_k, None)
                metadata[k] = item_metadata
            else:
                metadata[k] = v

    else: # If style is specified
        metadata['segmented'] = False
        pass # TBD

    # Add other metadata
    metadata['image_name'] = image_name
    metadata['image_path'] = os.path.join(image_dir, f"{image_name}.jpg")

    return metadata
    