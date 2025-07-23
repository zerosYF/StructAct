from typing import Dict, List, Any
from abc import ABC, abstractmethod
from search.config import SearchConfig
import random
class TaskBase(ABC):
    def __init__(self, config:SearchConfig):
        self.name = "BaseTask"
        self.config = config
        self.train_data_mcts = None
        self.train_data_rnn = None
        self.val_data = None
        self.test_data = None
        self.system_prompt = None

    def sample_train(self) -> List[Dict]:
        return random.sample(self.train_data_mcts, self.config.batch_size)
    
    def sample_train_rnn(self) -> List[Dict]:
        return random.sample(self.train_data_rnn, self.config.rnn_batch_size)
    
    def get_val(self) -> List[Dict]:
        return self.val_data
    
    def get_test(self) -> List[Dict]:
        return self.test_data
    
    @abstractmethod
    def inject_final_input(self, current_prompt:str, input:str) -> str:
        pass

    @abstractmethod
    def extract_origin_prompt(self) -> str:
        pass

    @abstractmethod
    def extract_tuple(self, sample:dict) -> tuple:
        pass

    @abstractmethod
    def samples2text(self, samples:List[Dict]) -> str:
        pass
    @abstractmethod
    def get_reward(self, output: Any, target: Any) -> float:
        pass