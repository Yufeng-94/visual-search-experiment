import torch

def post_process_bboxes(bboxes: torch.Tensor) -> torch.Tensor:
    '''
    Post-process bounding boxes to ensure they are in (x, y, w, h) format.
    
    Args:
        bboxes (torch.Tensor): A tensor of shape (4) representing bounding boxes.
                               Each box is in (x1, y1, x2, y2).
    Returns: A tensor of shape (4) in (x, y, w, h) format.
    '''
    if bboxes.numel() != 4:
        raise ValueError("Input tensor must have exactly 4 elements.")
    
    x1, y1, x2, y2 = bboxes.tolist()
    w = x2 - x1
    h = y2 - y1
    return torch.tensor([x1, y1, w, h])
