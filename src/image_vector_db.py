from abc import ABC, abstractmethod

class ImageVectorDB(ABC):
    @abstractmethod
    def add_images_batch(self):
        pass

    @abstractmethod
    def remove_images_batch(self):
        pass
    
    @abstractmethod
    def query_image(self):
        pass

    @abstractmethod
    def query_images_batch(self):
        pass
        