from typing import Dict, List, Any
from abc import ABC, abstractmethod
from search.config import SearchConfig
import random
class TaskBase(ABC):
    def __init__(self, config:SearchConfig):
        self.name = "BaseTask"
        self.config = config
        # train_data_all = train_data_mcts + eval_data_mcts
        self.train_data_rnn = None
        self.train_data_mcts = None
        self.eval_data_mcts = None
        self.test_data = None
        self.system_prompt = None
        self.answer_format_prompt = "At the end show the answer option bracketed between <answer> and </answer>."

    # train mcts actions sample mini_batch
    def sample_train(self) -> List[Dict]:
        return random.sample(self.train_data_mcts, self.config.batch_size)
    
    # train rnn structure
    def get_train(self) -> List[Dict]:
        return self.train_data_rnn 
    
    # get reward to update mcts q
    def get_eval(self) -> List[Dict]:
        return self.eval_data_mcts
    
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
    def _normalize_answer(self, text: str) -> str:
        """Normalize text by lowercasing, stripping, and removing punctuation."""
        pass

    @abstractmethod
    def get_reward(self, output: Any, target: Any) -> float:
        pass