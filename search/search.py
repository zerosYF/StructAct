from task.base_task import TaskBase
from src.action.base_action import OptimizeAction
from src.evaluator import PromptEvaluator
from src.config import SearchConfig
from typing import List, Set
from src.action.strategy_actions import define_full_actions
import math
from abc import abstractmethod, ABC

class SearchController(ABC):
    def __init__(self, 
                 evaluator: PromptEvaluator, 
                 config: SearchConfig, 
                 task: TaskBase):
        self.evaluator: PromptEvaluator = evaluator
        self.config: SearchConfig = config
        self.task: TaskBase = task
    
    @abstractmethod
    def search(self) ->tuple[str, str]:
        '''main search workflow'''
    
    def nonlinear_schedule(
        self,
        rnn_iter: int,
        total_iter: int,
        min_val: int,
        max_val: int,
        steepness: float = 10.0,
        pivot: float = 0.9
    ) -> int:
        """
        Nonlinear scheduling using a sigmoid curve.

        Args:
            rnn_iter (int): Current RNN iteration index.
            total_iter (int): Total RNN iterations.
            min_val (int): Minimum scheduled value.
            max_val (int): Maximum scheduled value.
            steepness (float): Controls curve steepness.
            pivot (float): Fraction [0,1] indicating where rapid increase starts.

        Returns:
            int: Scheduled integer value.
        """
        progress = rnn_iter / total_iter
        sigmoid = 1 / (1 + math.exp(-steepness * (progress - pivot)))
        value = min_val + (max_val - min_val) * sigmoid
        return max(min_val, min(max_val, int(round(value))))

    def schedule_mcts_iter(self, rnn_iter: int, total_iter: int, min_val=1, max_val=10) -> int:
        return self.nonlinear_schedule(rnn_iter, total_iter, min_val=min_val, max_val=max_val, steepness=12.0, pivot=0.993)

    def schedule_rollout_length(self, rnn_iter: int, total_iter: int, min_val=1, max_val=5) -> int:
        return self.nonlinear_schedule(rnn_iter, total_iter, min_val=min_val, max_val=max_val, steepness=12.0, pivot=0.993)

    def schedule_expand_num(self, rnn_iter: int, total_iter: int, min_val=0, max_val=3) -> int:
        return self.nonlinear_schedule(rnn_iter, total_iter, min_val=min_val, max_val=max_val, steepness=12.0, pivot=0.993)