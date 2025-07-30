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
            "###Requirement:\n"
            "This block specifies how the task objective is presented to the model.\n"
            f"You can present the task objective in a **{self.style}** manner here.\n"
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
            "###Requirement:\n"
            "This block defines the role or persona the model should assume.\n"
            "Roles influence style and expertise in responses.\n"
            f"Let model assume the role as a **{self.role_template}**.\n"
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
                "###Requirement:"
                "No expert knowledge should be provided here.\n"
                "</BLOCK:EXPERT_KNOWLEDGE>\n"
            )

        return (
            "<BLOCK:EXPERT_KNOWLEDGE>\n"
            "###Requirement:\n"
            "You should incorporate relevant domain-specific expert knowledge to enhance reasoning and accuracy.\n"
            "You are encouraged to derive appropriate theories, principles, or domain heuristics based on the task objective and provided information.\n"
            "This may include laws, standard practices, or expert-level insights relevant to the input task.\n"
            "</BLOCK:EXPERT_KNOWLEDGE>\n"
        )

class GuidanceBlock(PromptBlock):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.use_guidance = False  
        self.hyperparams = [0]    

    def name(self):
        return "Guidance"

    def get_search_space(self):
        return [2] 

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.use_guidance = bool(hyperparams[0])

    def describe(self):
        return {
            "type": "guidance",
            "enabled": self.use_guidance
        }

    def render(self):
        if not self.use_guidance:
            return (
                "<BLOCK:GUIDANCE>\n"
                "###Requirement:\n"
                "No reasoning guidance should be provided.\n"
                "</BLOCK:GUIDANCE>\n"
            )

        return (
            "<BLOCK:GUIDANCE>\n"
            "###Requirement:\n"
            "Some guidance information about this task you can insert here.\n"
            "You should use ​appropriate information context provided to fill this block, guide model perform the task more accurately.\n"
            "</BLOCK:GUIDANCE>\n"
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
                "###Requirement:\n"
                "**No example** should be provided here.\n"
                "Zero-shot is applied.\n"
                "</BLOCK:FEW_SHOT_EXAMPLES>\n"
            )
        
        if self.format == "Input-Output":
            return (
                "<BLOCK:FEW_SHOT_EXAMPLES>\n"
                "###Requirement:\n"
                f"This block should provide **{self.num}** example(s) with input and output pairs.\n"
                f"Example(s) are organized by **{self.order}**.\n"
                "Each example should consist of an input and an output.\n"
                "</BLOCK:FEW_SHOT_EXAMPLES>\n"
            )
        
        if self.format == "Input-Analysis-Output":
            return (
                "<BLOCK:FEW_SHOT_EXAMPLES>\n"
                "###Requirement:\n"
                f"This block should provide **{self.num}** example(s) with input, detailed step-by-step analysis, and output.\n"
                f"Example(s) are organized by **{self.order}**.\n"
                "Each example should consist of an input, an analysis of how to derive the output from the input, and the corresponding output.\n"
                "</BLOCK:FEW_SHOT_EXAMPLES>\n"
            )

class ConstraintBlock(PromptBlock):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.enable_options = [False, True]  
        self.format_options = ["Paragraph", "Bullet list"]
        self.hyperparams = [0, 0]  # [enable, format]
        self.enable = self.enable_options[self.hyperparams[0]]  
        self.format = self.format_options[self.hyperparams[1]]  

    def name(self): 
        return "ConstraintBlock"

    def get_search_space(self): 
        return [len(self.enable_options), len(self.format_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.enable = self.enable_options[hyperparams[0]]
        self.format = self.format_options[hyperparams[1]]

    def describe(self):
        return {
            "type": "constraint", 
            "enabled": self.enable, 
            "format": self.format
        }

    def render(self):
        if not self.enable:
            return (
                "<BLOCK:CONSTRAINTS>\n"
                "###Requirement:\n"
                "No additional constraints are enabled.\n"
                "</BLOCK:CONSTRAINTS>\n"
            )
        return (
            "<BLOCK:CONSTRAINTS>\n"
            "###Requirement:\n"
            "This block defines explicit constraints or rules the model must obey.\n"
            "You can use ​appropriate information I provided to fill this block.\n"
            f"Impose constraints formatted as **{self.format}**.\n"
            "</BLOCK:CONSTRAINTS>\n"
        )

class CautionBlock(PromptBlock):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.enable_options = [False, True]  
        self.style_options = ["Gentle reminder", "Strict directive"]
        self.hyperparams = [0, 0]  # [enable, style]
        self.enable = self.enable_options[self.hyperparams[0]]  
        self.style = self.style_options[self.hyperparams[1]]  

    def name(self): 
        return "CautionBlock"

    def get_search_space(self): 
        return [len(self.enable_options), len(self.style_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.enable = self.enable_options[hyperparams[0]]
        self.style = self.style_options[hyperparams[1]]

    def describe(self):
        return {
            "type": "caution", 
            "enabled": self.enable, 
            "style": self.style
        }

    def render(self):
        if not self.enable:
            return (
                "<BLOCK:CAUTIONS>\n"
                "###Requirement:\n"
                "No cautionary statements should be provided.\n"
                "</BLOCK:CAUTIONS>\n"
            )
        return (
            "<BLOCK:CAUTIONS>\n"
            "###Requirement:\n"
            "This block provides cautionary or warning statements to the model.\n"
            f"Include caution(s) styled as **{self.style}**.\n"
            "Content should be filled by analyzing information I provided.\n"
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
            "###Requirement:\n"
            "Specifies the format and structure of the model's answer output.\n"
            f"Derive model to answer in the following style: **{self.style}**.\n"
            "</BLOCK:ANSWER_STYLE>\n"
        )

def get_all_blocks(config:SearchConfig):
    return [
        TaskInstructionBlock(config),
        RoleBlock(config),
        ExpertKnowledgeBlock(config),
        GuidanceBlock(config),
        FewShotExampleBlock(config),
        ConstraintBlock(config),
        CautionBlock(config),
        AnswerStyleBlock(config),
    ]