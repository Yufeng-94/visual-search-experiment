import torch

def post_process_bboxes(bboxes: list) -> list:
    '''
    Post-process bounding boxes to ensure they are in (x, y, w, h) format.
    
    Args:
        bboxes (list): A tensor of shape (4) representing bounding boxes.
                               Each box is in (x1, y1, x2, y2).
    Returns: A list shape (4) in (x, y, w, h) format.
    '''
    if len(bboxes) != 4:
        raise ValueError("Input must have exactly 4 elements.")
    
    x1, y1, x2, y2 = bboxes
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]
