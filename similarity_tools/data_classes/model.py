from abc import ABC, abstractmethod
from typing import Optional

from bluegraph.downstream import EmbeddingPipeline


class Model(ABC):

    similarity: str
    dimension: int

    @abstractmethod
    def __init__(self, similarity, dimension):
        self.similarity = similarity
        self.dimension = dimension
        pass

    def run(self) -> Optional[EmbeddingPipeline]:
        pass
