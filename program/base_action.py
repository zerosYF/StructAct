from abc import ABC, abstractmethod
from model.model import Model, getEvalModel
from task.base_task import TaskBase
from logger import logger

class OptimizeAction(ABC):
    def __init__(self, task: TaskBase, name: str = None):
        self.name = name
        self.task = task
        self.usage_count = 0

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
        logger.info(f"📊 Current Template_description:\n{template_description}")

class StructureSyncAction(OptimizeAction):
    def __init__(self, task, name="StructureSyncAction"):
        super().__init__(task, name)
        self.rewriter_model: Model = getEvalModel()

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)

        # Sample examples for filling content (e.g., few-shot, constraints, etc.)
        samples = self.task.sample_train()

        # Construct few-shot example fragment (extract real QA pairs)
        example_texts = self.task.samples2text(samples)

        system_prompt = (
            "You are a prompt optimization assistant. Your task is to rewrite the prompt "
            "based on a structural template and sample information to align the content with the structure.\n"
            "Strictly follow the constraints of the structural template, and make minimal updates to the content.\n"
            "Maintain consistent style and accurate expression. Avoid redundancy."
        )

        rewrite_prompt = (
            f"Current Prompt:\n{current_prompt}\n\n"
            f"Structural Template Description:\n{template_description}\n\n"
            f"Reference Sample QA Pairs:\n{example_texts}\n\n"
            f"Please revise the current prompt based on the given structure, using the information in the examples, "
            f"to make the content aligned with the structure.\n"
            f"Do not alter the structural settings."
        )

        rewritten_prompt = self.rewriter_model.api_call(system_prompt, rewrite_prompt)
        return rewritten_prompt

    