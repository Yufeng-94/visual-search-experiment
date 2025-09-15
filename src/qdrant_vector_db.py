from image_vector_db import ImageVectorDB
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointIdsList
from typing import List
import json
import torch
import torchvision
import logging

from torchvision.io import decode_image

METADATA_ITEM_KEYS = ['scale', 'viewpoint', 'zoom_in', 'style', 'bounding_box', 'occlusion', 'category_id']

logging.basicConfig(level=logging.INFO)

class QdrantVectorDB(ImageVectorDB):
    """
    A concrete implementation of VectorDB using FAISS. The similarity metric is Inner Product.
    """
    def __init__(self):
        raise NotImplementedError("Use 'create_db' class method to create a new Qdrant collection.")

    @classmethod
    def init_client(cls, database_path: str) -> QdrantClient:
        client = QdrantClient(path=database_path)
        return client

    @classmethod
    def create_db(
        cls, 
        database_name: str, 
        dimension: int, 
        distance: str, 
        database_path: str,
        client: QdrantClient=None,
        ):

        if client is None:
            client = QdrantClient(path=database_path)
        try:
            client.recreate_collection(
                collection_name=database_name,
                vectors_config=VectorParams(
                    size=dimension, distance=distance
                    )
            )
            logging.info(f"Created Qdrant collection '{database_name}' with dimension {dimension} and distance '{distance}'.")

            instance = cls.__new__(cls)
            instance._initialize(client, database_name, dimension, distance)

            return instance

        except Exception as e:
            logging.error(f"Failed to create Qdrant collection '{database_name}': {e}")
            raise e
    
    @classmethod
    def load_db(cls, database_name: str, database_path: str, client: QdrantClient=None):
        if client is None:
            client = QdrantClient(path=database_path)
        try:
            collection_info = client.get_collection(collection_name=database_name)
            dimension = collection_info.config.params.vectors.size
            distance = collection_info.config.params.vectors.distance
            logging.info(f"Loaded Qdrant collection '{database_name}' with dimension {dimension} and distance '{distance}'.")

            instance = cls.__new__(cls)
            instance._initialize(client, database_name, dimension, distance)

            return instance
        
        except Exception as e:
            logging.error(f"Failed to load Qdrant collection '{database_name}': {e}")
            raise e
        
    def _initialize(self, client: QdrantClient, database_name: str, dimension: int, distance: str):
        self.client = client
        self.database_name = database_name
        self.dimension = dimension
        self.distance = distance


    def add_images_batch(
            self,
            image_paths: List[str], 
            metadata_paths: List[str], 
            image_encoder: torch.nn.Module, 
            image_transforms: torchvision.transforms,
            batch_size: int=128,
            parallel: int=4,
            device: torch.device=torch.device('cpu'),
            ):
        # get image paths by batch
        for i in range(0, len(image_paths), batch_size):
            try:
                batch_image_paths = image_paths[i: i+batch_size]
                batch_metadata_paths = metadata_paths[i: i+batch_size]

                # get image tensors
                batch_image_tensors = self._load_images(batch_image_paths, image_transforms)
                batch_image_tensors = batch_image_tensors.to(device)

                # get image metadata
                batch_metedata = self._load_metadata(batch_metadata_paths, batch_image_paths)
                batch_ids = [int(m['image_name']) for m in batch_metedata]

                # batch encode images
                with torch.no_grad():
                    batch_image_embeddings = image_encoder(batch_image_tensors).cpu().numpy()

                # add to Qdrant
                self.client.upload_collection(
                    collection_name=self.database_name,
                    vectors=batch_image_embeddings,
                    payload=batch_metedata,
                    ids=batch_ids,
                    parallel=parallel,
                )

                logging.info(f"Added {len(batch_image_paths)} images to Qdrant collection '{self.database_name}'.")

            except Exception as e:
                logging.error(f"Failed to add images to Qdrant collection '{self.database_name}': {e}")
                continue

    def _load_images(
            self, 
            image_paths: List[str], 
            transforms: torchvision.transforms
            ) -> torch.Tensor:
        
        image_tensors = []
        for i_p in image_paths:
            image_tensor = decode_image(i_p) # [3, H, W]
            image_tensor = transforms(image_tensor) # [3, H, W]
            image_tensors.append(image_tensor)

        return torch.stack(image_tensors, dim=0) # [batch_size, 3, H, W]

    def _load_metadata(self, metadata_paths: List[str], image_paths: List[str]) -> List[dict]:
        loaded_metadata = []
        for m_p, i_p in zip(metadata_paths, image_paths):
            # Load metadata JSON
            with open(m_p, 'r') as f:
                metadata_raw = json.load(f)

            # Prepare extracted metadata
            metadata = {}
            metadata['segmented'] = False
            for k, v in metadata_raw.items():
                if 'item' in k:
                    item_metadata = {}
                    for i_k in METADATA_ITEM_KEYS:
                        item_metadata[i_k] = v.get(i_k, None)
                    metadata[k] = item_metadata
                else:
                    metadata[k] = v

            # Add other metadata
            ## NOTE: could replace with Path
            metadata['image_name'] = m_p.split('/')[-1].replace('.json', '') 
            metadata['image_path'] = i_p

            loaded_metadata.append(metadata)

        return loaded_metadata

    def remove_images_batch(self, image_names: List[str]):
        image_ids = [int(name) for name in image_names]
        image_ids = PointIdsList(points=image_ids)
        try:
            response = self.client.delete(
                collection_name=self.database_name,
                points_selector=image_ids
            )
            if response.status == 'completed':
                logging.info(f"Removed images from Qdrant collection '{self.database_name}'.")
            else:
                logging.error(f"Failed to remove images from Qdrant collection '{self.database_name}': {response.status}")

        except Exception as e:
            logging.error(f"Failed to remove images from Qdrant collection '{self.database_name}': {e}")
    
    def query_image(self, query_vector: list, k: int=5) -> list:
        response = self.client.query_points(
            collection_name=self.database_name,
            query=query_vector,
            limit=k,
        )
        
        return response.points

    def query_images_batch(self):
        pass

    def get_number_of_vectors(self) -> int:
        try:
            collection_info = self.client.get_collection(collection_name=self.database_name)
            num_vectors = collection_info.points_count
            return num_vectors
        except Exception as e:
            logging.error(f"Failed to get number of vectors in Qdrant collection '{self.database_name}': {e}")
            return -1

    def disconnect(self):
        self.client.close()

    def get_client(self) -> QdrantClient:
        return self.client

    @classmethod
    def remove_db(cls, database_name: str, database_path: str, client: QdrantClient=None):
        if client is None:
            client = QdrantClient(path=database_path)
        try:
            client.delete_collection(collection_name=database_name)
            logging.info(f"Deleted Qdrant collect '{database_name}'.")
        except Exception as e:
            logging.error(f"Failed to delete Qdrant collection '{database_name}': {e}")

        