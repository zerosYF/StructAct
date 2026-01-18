from abc import ABC, abstractmethod
from task.base_task import TaskBase
from src.logger import logger

class OptimizeAction(ABC):
    def __init__(self, task: TaskBase, name: str = None, max_sample_num: int = 1):
        self.name = name
        self.task = task
        self.usage_count = 0
        self.sample_failure_count = 0
        self.max_sample_num = max_sample_num
        self.log_file = "logs/optimization_log.txt"

    @abstractmethod
    def do(self, 
           current_prompt: str, 
           trajectory_prompts: list[str] = None,  
           sample_pool=None) -> str:
        """
        Args:
            current_prompt: the current prompt string
        Returns:
            updated prompt string
        Note:
            samples: List[dict] like [{"input": ..., "output": ...}, ...]
            structure: current prompt structural template
        """
        self.usage_count += 1
        logger.info(f"📊 Current Prompt:\n{current_prompt}")


    