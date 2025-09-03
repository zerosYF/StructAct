from abc import ABC, abstractmethod
from typing import List
from search.config import SearchConfig

class PromptBlock(ABC):
    def __init__(self, config:SearchConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_search_space(self) -> List[int]:
        pass

    @abstractmethod
    def set_hyperparams(self, hyperparams: List[int]):
        pass

    @abstractmethod
    def render(self) -> dict:
        pass

    @abstractmethod
    def describe(self) -> str:
        pass

    def get_num_slots(self) -> int:
        return len(self.get_search_space())
    



