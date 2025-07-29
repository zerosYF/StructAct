from program.base_block import PromptBlock
from typing import List
from search.config import SearchConfig

class TaskInstructionBlock(PromptBlock):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.style_options = [
            "Concise",   
            "Detailed"   
        ]
        self.hyperparams = [0]
        self.style = self.style_options[self.hyperparams[0]]

    def name(self): 
        return "TaskInstruction"

    def get_search_space(self): 
        return [len(self.style_options)]

    def set_hyperparams(self, hyperparams: list[int]):
        self.hyperparams = hyperparams
        self.style = self.style_options[hyperparams[0]]

    def describe(self):
        return {"type": "task_instruction", "style": self.style}

    def render(self):
        return (
            "<BLOCK:TASK_INSTRUCTION>\n"
            "This block specifies how the task objective is presented to the model.\n"
            f"Present the task objective in a <STYLE={self.style}> manner.\n"
            "</BLOCK:TASK_INSTRUCTION>\n"
        )

class RoleBlock(PromptBlock):
    def __init__(self, config:SearchConfig):
        super().__init__(config)
        self.role_templates = [
            "Assistant (Neutral)",
            "Math Tutor (Formal)",
            "Logical Analyst (Contextualized)",
            "Data Interpreter (Neutral)",
            "Visual Coder (Formal)",
            "Engineer (Problem-Solver)",  
            "Business Expert (Strategic)",  
            "Ethical Leader (Responsible)"  
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
            "This block defines the role or persona the model should assume.\n"
            "Roles influence style and expertise in responses.\n"
            f"Assume the role as <ROLE_TEMPLATE={self.role_template}>.\n"
            "</BLOCK:ROLE>\n"
        )

class ExpertKnowledgeBlock(PromptBlock):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.use_knowledge = False  
        self.hyperparams = [0]     

    def name(self):
        return "ExpertKnowledge"

    def get_search_space(self):
        return [2]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.use_knowledge = bool(hyperparams[0])

    def describe(self):
        return {
            "type": "expert_knowledge",
            "enabled": self.use_knowledge
        }

    def render(self):
        if not self.use_knowledge:
            return (
                "<BLOCK:EXPERT_KNOWLEDGE>\n"
                "No expert knowledge can be provided.\n"
                "</BLOCK:EXPERT_KNOWLEDGE>\n"
            )

        return (
            "<BLOCK:EXPERT_KNOWLEDGE>\n"
            "The model should incorporate relevant domain-specific expert knowledge to enhance reasoning and accuracy.\n"
            "You are encouraged to derive appropriate theories, principles, or domain heuristics based on the task objective and provided examples.\n"
            "This may include laws, standard practices, or expert-level insights relevant to the input task.\n"
            "</BLOCK:EXPERT_KNOWLEDGE>\n"
        )

class FewShotExampleBlock(PromptBlock):
    def __init__(self, config:SearchConfig):
        super().__init__(config)
        self.num_options = [0, 3, 5]  # Options for the number of examples
        assert max(self.num_options) <= config.batch_size
        self.order_options = ["Random", "Semantic similarity"]
        self.format_options = ["Input-Output", "Input-Analysis-Output"]  # New format option
        self.hyperparams = [0, 0, 0]  # Includes format choice
        self.num = self.num_options[self.hyperparams[0]]
        self.order = self.order_options[self.hyperparams[1]]
        self.format = self.format_options[self.hyperparams[2]]

    def name(self): return "FewShotExamples"

    def get_search_space(self): 
        return [len(self.num_options), len(self.order_options), len(self.format_options)]  # Expanded search space

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.num = self.num_options[hyperparams[0]]
        self.order = self.order_options[hyperparams[1]]
        self.format = self.format_options[hyperparams[2]]

    def describe(self):
        return {
            "type": "few_shot_examples", 
            "num_examples": self.num, 
            "ordering": self.order, 
            "format": self.format
        }

    def render(self):
        if self.num == 0:
            return (
                "<BLOCK:FEW_SHOT_EXAMPLES>\n"
                "No few-shot examples can be provided.\n"
                "Zero-shot is applied.\n"
                "</BLOCK:FEW_SHOT_EXAMPLES>\n"
            )
        
        if self.format == "Input-Output":
            return (
                "<BLOCK:FEW_SHOT_EXAMPLES>\n"
                "This block includes examples with input and output pairs.\n"
                "Ordering affects how examples are selected.\n"
                f"Provide <NUM_EXAMPLES={self.num}> example(s), organized by <ORDERING={self.order}>.\n"
                "Each example should consist of an input and an output.\n"
                "</BLOCK:FEW_SHOT_EXAMPLES>\n"
            )
        
        if self.format == "Input-Analysis-Output":
            return (
                "<BLOCK:FEW_SHOT_EXAMPLES>\n"
                "This block includes examples with input, detailed analysis, and output.\n"
                "Ordering affects how examples are selected.\n"
                f"Provide <NUM_EXAMPLES={self.num}> example(s), organized by <ORDERING={self.order}>.\n"
                "Each example should consist of an input, an analysis of the input, and the corresponding output.\n"
                "</BLOCK:FEW_SHOT_EXAMPLES>\n"
            )

class ConstraintBlock(PromptBlock):
    def __init__(self, config:SearchConfig):
        super().__init__(config)
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
            return (
                "<BLOCK:CONSTRAINTS>\n"
                "No additional constraints can be applied.\n"
                "</BLOCK:CONSTRAINTS>\n"
            )
        return (
            "<BLOCK:CONSTRAINTS>\n"
            "This block defines explicit constraints or rules model must obey.\n"
            f"Impose <NUM_CONSTRAINTS={self.num}> constraint(s), formatted as <FORMAT={self.format}>.\n"
            "</BLOCK:CONSTRAINTS>\n"
        )

class CautionBlock(PromptBlock):
    def __init__(self, config:SearchConfig):
        super().__init__(config)
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
            return (
                "<BLOCK:CAUTIONS>\n"
                "No cautionary statements are needed.\n"
                "</BLOCK:CAUTIONS>\n"
            )
        return (
            "<BLOCK:CAUTIONS>\n"
            "This block provides cautionary or warning statements to the model.\n"
            f"Include <NUM_CAUTIONS={self.count}> caution(s) styled as <STYLE={self.style}>. "
            "Content is dynamically generated.\n"
            "</BLOCK:CAUTIONS>\n"
        )

class AnswerStyleBlock(PromptBlock):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.style_options = [
            "Final answer only",  
            "Chain-of-Thought reasoning with final answer",  
            "Step-by-step reasoning in bullet points",  
            "Explain-then-answer",  
        ]
        self.hyperparams = [0]
        self.style = self.style_options[self.hyperparams[0]]

    def name(self): return "AnswerStyle"

    def get_search_space(self): return [len(self.style_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.style = self.style_options[hyperparams[0]]

    def describe(self):
        return {"type": "answer_style", "style": self.style}

    def render(self):
        return (
            "<BLOCK:ANSWER_STYLE>\n"
            "Specifies the format and structure of the model's answer output.\n"
            f"Format your answer in the following style: <STYLE={self.style}>.\n"
            "</BLOCK:ANSWER_STYLE>\n"
        )

def get_all_blocks(config:SearchConfig):
    return [
        TaskInstructionBlock(config),
        RoleBlock(config),
        ExpertKnowledgeBlock(config),
        FewShotExampleBlock(config),
        ConstraintBlock(config),
        CautionBlock(config),
        AnswerStyleBlock(config),
    ]