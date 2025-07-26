from program.base_block import PromptBlock
from typing import List

class TaskInstructionBlock(PromptBlock):
    def __init__(self):
        self.style_options = [
            "Concise objective only",
            "Encouraging formal objective with Chain-of-Thought",
            "Encouraging tone with detailed reasoning"
        ]
        self.hyperparams = [0]
        self.style = self.style_options[self.hyperparams[0]]

    def name(self): return "TaskInstruction"

    def get_search_space(self): return [len(self.style_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.style = self.style_options[hyperparams[0]]

    def describe(self):
        return {"type": "task_instruction", "style": self.style}

    def render(self):
        return (
            "<BLOCK:TASK_INSTRUCTION>\n"
            f"State the task objective using the following style: <STYLE={self.style}>.\n"
            "</BLOCK:TASK_INSTRUCTION>\n"
        )

class RoleBlock(PromptBlock):
    def __init__(self):
        self.role_templates = [
            "Assistant (Neutral)",
            "Math Tutor (Formal)",
            "Logical Analyst (Contextualized)",
            "Data Interpreter (Neutral)",
            "Visual Coder (Formal)"
        ]
        self.hyperparams = [0]
        self.role_template = self.role_templates[self.hyperparams[0]]

    def name(self): return "Role"

    def get_search_space(self): return [len(self.role_templates)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.role_template = self.role_templates[hyperparams[0]]

    def describe(self):
        return {"type": "role", "role_template": self.role_template}

    def render(self):
        return (
            "<BLOCK:ROLE>\n"
            f"Assume the role as <ROLE_TEMPLATE={self.role_template}>.\n"
            "</BLOCK:ROLE>\n"
        )

class FewShotExampleBlock(PromptBlock):
    def __init__(self):
        self.num_options = [0, 3, 7]
        self.order_options = ["Random", "Semantic similarity"]
        self.hyperparams = [0, 0]
        self.num = self.num_options[self.hyperparams[0]]
        self.order = self.order_options[self.hyperparams[1]]

    def name(self): return "FewShotExamples"

    def get_search_space(self): return [len(self.num_options), len(self.order_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.num = self.num_options[hyperparams[0]]
        self.order = self.order_options[hyperparams[1]]

    def describe(self):
        return {"type": "few_shot_examples", "num_examples": self.num, "ordering": self.order}

    def render(self):
        if self.num == 0:
            return "<BLOCK:FEW_SHOT_EXAMPLES>\nNo few-shot examples are provided.\n</BLOCK:FEW_SHOT_EXAMPLES>\n"
        return (
            "<BLOCK:FEW_SHOT_EXAMPLES>\n"
            f"Provide <NUM_EXAMPLES={self.num}> example(s), organized by <ORDERING={self.order}>.\n"
            "</BLOCK:FEW_SHOT_EXAMPLES>\n"
        )

class ConstraintBlock(PromptBlock):
    def __init__(self):
        self.num_options = [0, 3, 7]
        self.format_options = ["Paragraph", "Bullet list"]
        self.hyperparams = [0, 0]
        self.num = self.num_options[self.hyperparams[0]]
        self.format = self.format_options[self.hyperparams[1]]

    def name(self): return "ConstraintBlock"

    def get_search_space(self): return [len(self.num_options), len(self.format_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.num = self.num_options[hyperparams[0]]
        self.format = self.format_options[hyperparams[1]]

    def describe(self):
        return {"type": "constraint", "num_constraints": self.num, "format": self.format}

    def render(self):
        if self.num == 0:
            return "<BLOCK:CONSTRAINTS>\nNo additional constraints are applied.\n</BLOCK:CONSTRAINTS>\n"
        return (
            "<BLOCK:CONSTRAINTS>\n"
            f"Impose <NUM_CONSTRAINTS={self.num}> constraint(s), formatted as <FORMAT={self.format}>.\n"
            "</BLOCK:CONSTRAINTS>\n"
        )

class CautionBlock(PromptBlock):
    def __init__(self):
        self.count_options = [0, 5, 7]
        self.style_options = ["Gentle reminder", "Strict directive"]
        self.hyperparams = [0, 0]
        self.count = self.count_options[self.hyperparams[0]]
        self.style = self.style_options[self.hyperparams[1]]

    def name(self): return "CautionBlock"

    def get_search_space(self): return [len(self.count_options), len(self.style_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.count = self.count_options[hyperparams[0]]
        self.style = self.style_options[hyperparams[1]]

    def describe(self):
        return {"type": "caution", "count": self.count, "style": self.style}

    def render(self):
        if self.count == 0:
            return "<BLOCK:CAUTIONS>\nNo cautionary statements are needed.\n</BLOCK:CAUTIONS>\n"
        return (
            "<BLOCK:CAUTIONS>\n"
            f"Include <NUM_CAUTIONS={self.count}> caution(s) styled as <STYLE={self.style}>. "
            "Content is dynamically generated.\n"
            "</BLOCK:CAUTIONS>\n"
        )

def get_all_blocks():
    return [
        TaskInstructionBlock(),
        RoleBlock(),
        FewShotExampleBlock(),
        ConstraintBlock(),
        CautionBlock(),
    ]