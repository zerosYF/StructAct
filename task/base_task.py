from typing import Dict, List
from abc import ABC, abstractmethod
from search.config import SearchConfig
import random
class TaskBase(ABC):
    def __init__(self, config:SearchConfig):
        self.config = config
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.system_prompt = None

    def sample_train(self) -> List[Dict]:
        return random.sample(self.train_data, self.config.batch_size)
    
    def get_val(self):
        return self.val_data
    
    def get_test(self):
        return self.test_data
    
    @abstractmethod
    def inject_final_input(self, current_prompt:str, input:str) -> str:
        pass

    @abstractmethod
    def extract_origin_prompt(self) -> str:
        pass

    @abstractmethod
    def extract_tuple(self, sample:dict):
        pass