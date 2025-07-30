from abc import ABC, abstractmethod
from model.model import Model, getEvalModel
from task.base_task import TaskBase
import logging 

class OptimizeAction(ABC):
    def __init__(self, task: TaskBase, name: str = None):
        self.name = name
        self.task = task
        self.usage_count = 0
        self.log_file = "logs/optimization_log.txt"
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger(self.name or "OptimizeAction")
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

    @abstractmethod
    def do(self, 
           current_prompt: str, 
           template_description: str) -> str:
        """
        Args:
            current_prompt: the current prompt string
            template_description: the structural template description
        Returns:
            updated prompt string
        Note:
            samples: List[dict] like [{"input": ..., "output": ...}, ...]
            structure: current prompt structural template
        """
        self.usage_count += 1
        self.logger.info(f"📊 Current Template_description:\n{template_description}")
        self.logger.info(f"📊 Current Prompt:\n{current_prompt}")


class StructureSyncAction(OptimizeAction):
    def __init__(self, task, name="StructureSyncAction"):
        super().__init__(task, name)
        self.rewriter_model: Model = getEvalModel()

    def do(self, current_prompt, template_description):

        # Sample examples for filling content (e.g., few-shot, constraints, etc.)
        samples = self.task.sample_train_mcts(self.task.config.batch_size)

        # Construct few-shot example fragment (extract real QA pairs)
        example_texts = self.task.samples2text(samples)

        rewrite_prompt = (
            "I'm writing a prompt for a language model designed for a task.\n\n"
            f"My current prompt:\n{current_prompt}\n\n"
            f"My template  description:\n{template_description}\n\n"
            "Your revision must strictly follow my prompt template description.\n\n"
            f"There are some examples QA pairs you can use to get information:\n{example_texts}\n\n"
            f"Please help me revise my current prompt based on the given template description.\n\n"
            f"You must \n\n"
            f"Do not alter the template settings.\n\n"
            f"Just output revise prompt without other text.\n"
        )

        rewritten_prompt = self.rewriter_model.api_call(rewrite_prompt)
        super().do(rewritten_prompt, template_description)
        return rewritten_prompt

    