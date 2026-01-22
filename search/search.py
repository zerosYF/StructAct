from task.base_task import TaskBase
from src.evaluator import PromptEvaluator
from src.config import SearchConfig
from abc import abstractmethod, ABC
import math

class SearchController(ABC):
    def __init__(self, 
                 evaluator: PromptEvaluator, 
                 config: SearchConfig, 
                 task: TaskBase):
        self.evaluator: PromptEvaluator = evaluator
        self.config: SearchConfig = config
        self.task: TaskBase = task
    
    @abstractmethod
    def search(self) ->str:
        '''main search workflow'''