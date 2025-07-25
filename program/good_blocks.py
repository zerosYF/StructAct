from program.base_block import PromptBlock
from typing import List
class TaskObjectiveBlock(PromptBlock):
    def __init__(self):
        self.tone_options = ["Concise", "Formal", "Encouraging"]
        self.structure_options = ["None", "Weak hint", "Strong hint"]
        self.hyperparams = [0, 0]

        self.tone = self.tone_options[self.hyperparams[0]]
        self.structure = self.structure_options[self.hyperparams[1]]

    def name(self): return "TaskObjective"

    def get_search_space(self):
        return [len(self.tone_options), len(self.structure_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.tone = self.tone_options[hyperparams[0]]
        self.structure = self.structure_options[hyperparams[1]]

    def render(self):
        return {
            "type": "task_objective",
            "tone": self.tone,
            "structure_hint": self.structure
        }

    def describe(self):
        return (
            f"Block: TaskObjective - Tone: “{self.tone}”, Structure: “{self.structure}”."
        )

class RoleConstraintBlock(PromptBlock):
    def __init__(self):
        self.role_options = self.role_options = [
            "Assistant",         
            "Math Tutor",        
            "Logical Analyst",  
            "Data Interpreter",  
            "Visual Coder"      
        ]
        self.tone_options = ["Neutral", "Formal"]
        self.detail_options = ["Identity only", "Identity + full context"]
        self.hyperparams = [0, 0, 0]

        self.role = self.role_options[self.hyperparams[0]]
        self.tone = self.tone_options[self.hyperparams[1]]
        self.detail = self.detail_options[self.hyperparams[2]]

    def name(self): return "RoleConstraint"

    def get_search_space(self):
        return [len(self.role_options), len(self.tone_options), len(self.detail_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.role = self.role_options[hyperparams[0]]
        self.tone = self.tone_options[hyperparams[1]]
        self.detail = self.detail_options[hyperparams[2]]

    def render(self):
        return {
            "type": "role_constraint",
            "role": self.role,
            "tone": self.tone,
            "description_detail": self.detail
        }

    def describe(self):
        return (
            f"Block: RoleConstraint - Role: “{self.role}”, Tone: “{self.tone}”, Detail: “{self.detail}”."
        )

class FewShotExampleBlock(PromptBlock):
    def __init__(self):
        self.num_options = [0, 1, 3, 5]
        self.order_options = ["Random", "Semantic similarity"]
        self.hyperparams = [0, 0]

        self.num = self.num_options[self.hyperparams[0]]
        self.order = self.order_options[self.hyperparams[1]]

    def name(self): return "FewShotExamples"

    def get_search_space(self):
        return [len(self.num_options), len(self.order_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.num = self.num_options[hyperparams[0]]
        self.order = self.order_options[hyperparams[1]]

    def render(self):
        return {
            "type": "few_shot_examples",
            "num_examples": self.num,
            "ordering": self.order
        }

    def describe(self):
        return (
            f"Block: FewShotExamples - “{self.num}” example(s), Order: “{self.order}”."
        )

class ConstraintBlock(PromptBlock):
    def __init__(self):
        self.num_options = [0, 1, 3, 5]
        self.format_options = ["Paragraph", "Bullet list", "Numbered list"]
        self.hyperparams = [0, 0]

        self.num = self.num_options[self.hyperparams[0]]
        self.format = self.format_options[self.hyperparams[1]]

    def name(self): return "ConstraintBlock"

    def get_search_space(self):
        return [len(self.num_options), len(self.format_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.num = self.num_options[hyperparams[0]]
        self.format = self.format_options[hyperparams[1]]

    def render(self):
        return {
            "type": "constraint",
            "num_constraints": self.num,
            "format": self.format
        }

    def describe(self):
        return (
            f"Block: ConstraintBlock - “{self.num}” constraint(s), Format: “{self.format}”."
        )

class CautionBlock(PromptBlock):
    def __init__(self):
        self.count_options = [0, 5, 10]
        self.style_options = ["Gentle reminder", "Strict directive"]
        self.hyperparams = [0, 0]

        self.count = self.count_options[self.hyperparams[0]]
        self.style = self.style_options[self.hyperparams[1]]

    def name(self): return "CautionBlock"

    def get_search_space(self):
        return [len(self.count_options), len(self.style_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.style = self.style_options[hyperparams[0]]
        self.count = self.count_options[hyperparams[1]]

    def render(self):
        return {
            "type": "caution",
            "count": self.count,
            "style": self.style,
            "content_generation": "model_sampled"
        }

    def describe(self):
        return (
            f"Block: CautionBlock - “{self.count}” warning(s), Style: “{self.style}”."
        )

class SummaryClosureBlock(PromptBlock):
    def __init__(self):
        self.options = ["None", "Brief summary"]
        self.hyperparams = [0]

        self.summary_type = self.options[self.hyperparams[0]]

    def name(self): return "SummaryClosure"

    def get_search_space(self):
        return [len(self.options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.summary_type = self.options[hyperparams[0]]

    def render(self):
        return {
            "type": "summary_closure",
            "summary_type": self.summary_type
        }

    def describe(self):
        return (
            f"Block: SummaryClosure - Type: “{self.summary_type}”."
        )

class ReasoningStrategyBlock(PromptBlock):
    def __init__(self):
        self.strategy_options = ["None", "Chain of Thought"]
        self.verbosity_options = ["Concise", "Detailed"]
        self.hyperparams = [0, 0]

        self.strategy = self.strategy_options[self.hyperparams[0]]
        self.verbosity = self.verbosity_options[self.hyperparams[1]]

    def name(self): return "ReasoningStrategy"

    def get_search_space(self):
        return [len(self.strategy_options), len(self.verbosity_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.strategy = self.strategy_options[hyperparams[0]]
        self.verbosity = self.verbosity_options[hyperparams[1]]

    def render(self):
        return {
            "type": "reasoning_strategy",
            "strategy": self.strategy,
            "verbosity": self.verbosity
        }

    def describe(self):
        return (
            f"Block: ReasoningStrategy - Strategy: “{self.strategy}”, Verbosity: “{self.verbosity}”."
        )

def get_all_blocks():
    return [
        TaskObjectiveBlock(),
        RoleConstraintBlock(),
        FewShotExampleBlock(),
        ConstraintBlock(),
        CautionBlock(),
        SummaryClosureBlock(),
        ReasoningStrategyBlock(),
    ]