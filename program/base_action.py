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

        self.system_prompt = (
            "You are a prompt optimization assistant.\n"
            "Your task is to rewrite the prompt content to strictly align with the given structural template.\n"
            "Use the provided example QA pairs as references for content.\n"
            "Make only minimal necessary updates to ensure the prompt matches the template requirements.\n"
            "Maintain consistent style and accurate expression.\n"
            "Avoid redundancy or unnecessary additions.\n"
            "Do NOT alter the template structure or section order.\n"
            "Output only the fully revised prompt, without explanations or extra text."
        )

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)

        samples = self.task.sample_train_mcts(self.task.config.batch_size)
        example_texts = self.task.samples2text(samples)

        rewrite_prompt = (
            "<CurrentPrompt>\n"
            f"{current_prompt}\n"
            "</CurrentPrompt>\n\n"

            "<TemplateStructure>\n"
            f"{template_description}\n"
            "</TemplateStructure>\n\n"

            "<ReferenceExamples>\n"
            f"{example_texts}\n"
            "</ReferenceExamples>\n\n"

            "<Instruction>\n"
            "- Revise the prompt in <CurrentPrompt> to fully comply with <TemplateStructure>.\n"
            "- Use <ReferenceExamples> to adjust or enrich prompt content where needed.\n"
            "- Make minimal edits only, preserving style and accuracy.\n"
            "- Do NOT modify structure, section order, or formatting.\n"
            "- Avoid redundancy and unrelated additions.\n"
            "- Output only the revised full prompt, with no extra text.\n"
            "</Instruction>"
        )

        rewritten_prompt = self.rewriter_model.api_call(self.system_prompt, rewrite_prompt)
        return rewritten_prompt

    