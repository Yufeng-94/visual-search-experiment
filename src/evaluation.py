import torch
import qdrant_client

def mean_average_precision(relevant_scores: torch.Tensor) -> float:
    """
    Calculate Mean Average Precision (mAP) for a batch of queries.
    Args:
        relevant_scores (torch.Tensor): A binary tensor of shape (num_queries, num_items)
                                        where 1 indicates a relevant item and 0 indicates a non-relevant item.
    Returns:
        float: The mean average precision score.
    """
    if relevant_scores.dim == 1:
        relevant_scores = relevant_scores.unsqueeze(0)

    ranks = torch.arange(1, relevant_scores.shape[1] + 1)
    precision_at_k = torch.cumsum(relevant_scores, axis=1) / ranks
    average_precisions = torch.sum(precision_at_k * relevant_scores, axis=1) / torch.sum(relevant_scores, axis=1)
    average_precisions = torch.nan_to_num(average_precisions, nan=0.0, posinf=0.0, neginf=0.0)
    mean_ap = torch.mean(average_precisions)

    return mean_ap

def extract_retrieved_point_data(
        point: qdrant_client.models.ScoredPoint, extract_metadata: dict = None
        ) -> dict:
    """
    Extract relevant data from a Qdrant ScoredPoint object.
    Args:
        point (qdrant_client.models.ScoredPoint): The scored point object from Qdrant.
    Returns:
        dict: A dictionary containing the point's id, score, and metadata.
    """
    data = {
        'point_id': point.id,
        'score': point.score,
        'pair_id': point.payload['pair_id'],
        'image_name': point.payload['image_name'],
    }

    if extract_metadata:
        data.update(extract_metadata)

    return data