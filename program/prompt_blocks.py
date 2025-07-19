from program.base_block import PromptBlock
from typing import List

# 1. TaskObjectiveBlock
class TaskObjectiveBlock(PromptBlock):
    def __init__(self):
        self.tone_options = ["Concise", "Formal", "Encouraging", "Commanding"]
        self.verb_options = ["No verbs", "Use verbs", "Verb + adverb combination"]
        self.structure_options = ["No structure hint", "Include structure hint"]
        self.hyperparams = [0, 0, 0]

        self.tone = self.tone_options[self.hyperparams[0]]
        self.verb = self.verb_options[self.hyperparams[1]]
        self.structure = self.structure_options[self.hyperparams[2]]

    def name(self): return "TaskObjective"

    def get_search_space(self): 
        return [len(self.tone_options), len(self.verb_options), len(self.structure_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.tone = self.tone_options[hyperparams[0]]
        self.verb = self.verb_options[hyperparams[1]]
        self.structure = self.structure_options[hyperparams[2]]

    def render(self):
        return {
            "type": "task_objective",
            "tone": self.tone,
            "verb_usage": self.verb,
            "structure_hint": self.structure
        }

    def describe(self):
        return (
            f"Block: TaskObjective - This section should be written in a “{self.tone}” tone, "
            f"with {self.verb.lower()}, and {self.structure.lower()}."
        )

# 2. RoleConstraintBlock
class RoleConstraintBlock(PromptBlock):
    def __init__(self):
        self.role_options = ["General Assistant", "Domain Expert", "Teacher", "Code Generator", "Critical Analyst"]
        self.tone_options = ["Neutral", "Friendly", "Authoritative"]
        self.stance_options = ["Guide the user", "Obey instructions"]
        self.detail_options = ["Brief identity", "Identity + responsibilities", "Identity + responsibilities + communication style"]
        self.hyperparams = [0, 0, 0, 0]

        self.role = self.role_options[self.hyperparams[0]]
        self.tone = self.tone_options[self.hyperparams[1]]
        self.stance = self.stance_options[self.hyperparams[2]]
        self.detail = self.detail_options[self.hyperparams[3]]

    def name(self): return "RoleConstraint"

    def get_search_space(self): 
        return [len(self.role_options), len(self.tone_options), len(self.stance_options), len(self.detail_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.role = self.role_options[hyperparams[0]]
        self.tone = self.tone_options[hyperparams[1]]
        self.stance = self.stance_options[hyperparams[2]]
        self.detail = self.detail_options[hyperparams[3]]

    def render(self):
        return {
            "type": "role_constraint",
            "role": self.role,
            "tone": self.tone,
            "stance": self.stance,
            "description_detail": self.detail
        }

    def describe(self):
        return (
            f"Block: RoleConstraint - The model will act as a “{self.role}”, using a “{self.tone}” tone, "
            f"{'guiding the user' if self.stance == 'Guide the user' else 'obeying instructions'}, "
            f"with a role description of “{self.detail}”."
        )

# 3. FewShotExampleBlock
class FewShotExampleBlock(PromptBlock):
    def __init__(self):
        self.num_options = [0, 1, 2, 3, 4, 5]
        self.order_options = ["Random order", "Increasing difficulty", "Semantic similarity order"]
        self.content_options = ["Show full QA", "Only show answers"]
        self.hyperparams = [0, 0, 0]

        self.num = self.num_options[self.hyperparams[0]]
        self.order = self.order_options[self.hyperparams[1]]
        self.content = self.content_options[self.hyperparams[2]]

    def name(self): return "FewShotExamples"

    def get_search_space(self): 
        return [len(self.num_options), len(self.order_options), len(self.content_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.num = self.num_options[hyperparams[0]]
        self.order = self.order_options[hyperparams[1]]
        self.content = self.content_options[hyperparams[2]]

    def render(self):
        return {
            "type": "few_shot_examples",
            "num_examples": self.num,
            "ordering": self.order,
            "content_display": self.content
        }

    def describe(self):
        return (
            f"Block: FewShotExamples - Provides {self.num} example(s), ordered by {self.order.lower()}, "
            f"displaying: {self.content.lower()}."
        )

# 4. ConstraintBlock
class ConstraintBlock(PromptBlock):
    def __init__(self):
        self.num_options = [0, 1, 2, 3, 4, 5]
        self.style_options = ["Concise", "Detailed"]
        self.format_options = ["Paragraph", "List"]
        self.hyperparams = [0, 0, 0]

        self.num_constraints = self.num_options[self.hyperparams[0]]
        self.style = self.style_options[self.hyperparams[1]]
        self.format_style = self.format_options[self.hyperparams[2]]

    def name(self): return "ConstraintBlock"

    def get_search_space(self): 
        return [len(self.num_options), len(self.style_options), len(self.format_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.num_constraints = self.num_options[hyperparams[0]]
        self.style = self.style_options[hyperparams[1]]
        self.format_style = self.format_options[hyperparams[2]]

    def render(self):
        return {
            "type": "constraint",
            "num_constraints": self.num_constraints,
            "style": self.style,
            "format": self.format_style
        }

    def describe(self):
        return (
            f"Block: ConstraintBlock - Contains {self.num_constraints} constraint(s), "
            f"written in a “{self.style}” style and displayed in {self.format_style.lower()} format."
        )

# 5. CautionBlock
class CautionBlock(PromptBlock):
    def __init__(self):
        self.type_options = ["Factual accuracy", "Safety risk", "Moral boundaries", "No speculation", "Avoid repetition"]
        self.style_options = ["Gentle reminder", "Clear warning", "Brief note"]
        self.position_options = ["Before the task", "After the task"]
        self.count_options = [0, 1, 2, 3, 4]
        self.hyperparams = [0, 0, 0, 0]

        self.caution_type = self.type_options[self.hyperparams[0]]
        self.style = self.style_options[self.hyperparams[1]]
        self.position = self.position_options[self.hyperparams[2]]
        self.count = self.count_options[self.hyperparams[3]]

    def name(self): return "CautionBlock"

    def get_search_space(self): 
        return [len(self.type_options), len(self.style_options), len(self.position_options), len(self.count_options)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.caution_type = self.type_options[hyperparams[0]]
        self.style = self.style_options[hyperparams[1]]
        self.position = self.position_options[hyperparams[2]]
        self.count = self.count_options[hyperparams[3]]

    def render(self):
        return {
            "type": "caution",
            "caution_type": self.caution_type,
            "style": self.style,
            "position": self.position,
            "count": self.count
        }

    def describe(self):
        return (
            f"Block: CautionBlock - Provides “{self.caution_type}” cautions, styled as “{self.style}”, "
            f"placed {self.position.lower()}, with {self.count} total warning(s)."
        )

# 6. SummaryClosureBlock
class SummaryClosureBlock(PromptBlock):
    def __init__(self):
        self.include_summary = ["No summary", "Include summary"]
        self.include_next_step = ["No next step", "Include next step suggestion"]
        self.hyperparams = [0, 0]

        self.summary = self.include_summary[self.hyperparams[0]]
        self.next_step = self.include_next_step[self.hyperparams[1]]

    def name(self): return "SummaryClosure"

    def get_search_space(self): 
        return [len(self.include_summary), len(self.include_next_step)]

    def set_hyperparams(self, hyperparams: List[int]):
        self.hyperparams = hyperparams
        self.summary = self.include_summary[hyperparams[0]]
        self.next_step = self.include_next_step[hyperparams[1]]

    def render(self):
        return {
            "type": "summary_closure",
            "include_summary": self.summary,
            "include_next_step": self.next_step
        }

    def describe(self):
        return (
            f"Block: SummaryClosure - Includes summary: {self.summary.lower()}, "
            f"and next step: {self.next_step.lower()}."
        )

# Utility to get all blocks
def get_all_blocks():
    return [
        TaskObjectiveBlock(),
        RoleConstraintBlock(),
        FewShotExampleBlock(),
        ConstraintBlock(),
        CautionBlock(),
        SummaryClosureBlock(),
    ]