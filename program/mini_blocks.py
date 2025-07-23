from program.base_block import PromptBlock
from typing import List

# 1. TaskObjectiveBlock
class TaskObjectiveBlock(PromptBlock):
    def __init__(self):
        self.tone_options = ["Concise", "Formal", "Encouraging", "Commanding"]
        self.verb_options = ["No verbs", "Use verbs", "Verb + adverb combination"]
        self.hyperparams = [0, 0]

        self.tone = self.tone_options[self.hyperparams[0]]
        self.verb = self.verb_options[self.hyperparams[1]]

    def name(self): return "TaskObjective"

    def get_search_space(self): 
        return [len(self.tone_options), len(self.verb_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.tone = self.tone_options[hyperparams[0]]
        self.verb = self.verb_options[hyperparams[1]]

    def render(self):
        return {
            "type": "task_objective",
            "tone": self.tone,
            "verb_usage": self.verb,
        }

    def describe(self):
        return (
            f"Block: TaskObjective - This section should be written in a “{self.tone}” tone, "
            f"with style:“{self.verb.lower()}”."
        )

# 2. RoleConstraintBlock
class RoleConstraintBlock(PromptBlock):
    def __init__(self):
        self.role_options = ["General Assistant", "Domain Expert", "Teacher", "Code Generator", "Critical Analyst"]
        self.tone_options = ["Neutral", "Friendly", "Authoritative"]
        self.hyperparams = [0, 0]

        self.role = self.role_options[self.hyperparams[0]]
        self.tone = self.tone_options[self.hyperparams[1]]

    def name(self): return "RoleConstraint"

    def get_search_space(self): 
        return [len(self.role_options), len(self.tone_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.role = self.role_options[hyperparams[0]]
        self.tone = self.tone_options[hyperparams[1]]

    def render(self):
        return {
            "type": "role_constraint",
            "role": self.role,
            "tone": self.tone,
        }

    def describe(self):
        return (
            f"Block: RoleConstraint - The model will act as a “{self.role}”, using a “{self.tone}” tone, "
        )

# 3. FewShotExampleBlock
class FewShotExampleBlock(PromptBlock):
    def __init__(self):
        self.num_options = [0, 3, 5, 7]
        self.hyperparams = [0]

        self.num = self.num_options[self.hyperparams[0]]

    def name(self): return "FewShotExamples"

    def get_search_space(self): 
        return [len(self.num_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.num = self.num_options[hyperparams[0]]

    def render(self):
        return {
            "type": "few_shot_examples",
            "num_examples": self.num,
        }

    def describe(self):
        return (
            f"Block: FewShotExamples - Provides “{self.num}” example(s)."
        )

# 4. ConstraintBlock
class ConstraintBlock(PromptBlock):
    def __init__(self):
        self.num_options = [0, 3, 5, 7]
        self.format_options = ["Paragraph", "List"]
        self.hyperparams = [0, 0]

        self.num_constraints = self.num_options[self.hyperparams[0]]
        self.format_style = self.format_options[self.hyperparams[1]]

    def name(self): return "ConstraintBlock"

    def get_search_space(self): 
        return [len(self.num_options), len(self.format_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.num_constraints = self.num_options[hyperparams[0]]
        self.format_style = self.format_options[hyperparams[1]]

    def render(self):
        return {
            "type": "constraint",
            "num_constraints": self.num_constraints,
            "format": self.format_style
        }

    def describe(self):
        return (
            f"Block: ConstraintBlock - Contains “{self.num_constraints}” constraint(s), "
            f"displayed in “{self.format_style.lower()}” format."
        )

# 5. CautionBlock
class CautionBlock(PromptBlock):
    def __init__(self):
        self.style_options = ["Gentle reminder", "Clear warning", "Brief note"]
        self.count_options = [0, 3, 5, 7]
        self.hyperparams = [0, 0]

        self.style = self.style_options[self.hyperparams[0]]
        self.count = self.count_options[self.hyperparams[1]]

    def name(self): return "CautionBlock"

    def get_search_space(self): 
        return [len(self.style_options), len(self.count_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.style = self.style_options[hyperparams[0]]
        self.count = self.count_options[hyperparams[1]]

    def render(self):
        return {
            "type": "caution",
            "style": self.style,
            "count": self.count
        }

    def describe(self):
        return (
            f"Block: CautionBlock - Provides “{self.count}” total warning(s),"
            f"styled as “{self.style}”."
        )

# 6. SummaryClosureBlock
class SummaryClosureBlock(PromptBlock):
    def __init__(self):
        self.include_summary = ["No summary", "Include summary"]
        self.hyperparams = [0]

        self.summary = self.include_summary[self.hyperparams[0]]

    def name(self): return "SummaryClosure"

    def get_search_space(self): 
        return [len(self.include_summary)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.summary = self.include_summary[hyperparams[0]]

    def render(self):
        return {
            "type": "summary_closure",
            "include_summary": self.summary,
        }

    def describe(self):
        return (
            f"Block: SummaryClosure - Includes summary: “{self.summary.lower()}”."
        )

def get_all_blocks():
    return [
        TaskObjectiveBlock(),
        RoleConstraintBlock(),
        FewShotExampleBlock(),
        ConstraintBlock(),
        CautionBlock(),
        SummaryClosureBlock(),
    ]