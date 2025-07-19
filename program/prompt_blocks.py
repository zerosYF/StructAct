from program.base_block import PromptBlock
from typing import List
# 1. TaskObjectiveBlock
class TaskObjectiveBlock(PromptBlock):
    def __init__(self):
        self.tone_options = ["简洁", "正式", "鼓励", "命令"]
        self.verb_options = ["不使用动词", "使用动词", "动词+副词组合"]
        self.structure_options = ["不包含结构提示", "包含结构提示"]
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
        return f"该任务目标部分要求以「{self.tone}」语气书写，{self.verb}，且{self.structure}。"

# 2. RoleConstraintBlock
class RoleConstraintBlock(PromptBlock):
    def __init__(self):
        self.role_options = ["通用助手", "领域专家", "教师", "代码生成器", "批判性分析者"]
        self.tone_options = ["中性", "亲和", "权威"]
        self.stance_options = ["引导用户", "服从指令"]
        self.detail_options = ["简略身份", "身份+职责", "身份+职责+交流风格"]
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
            f"模型将作为“{self.role}”执行任务，语气为“{self.tone}”，"
            f"{'引导用户' if self.stance == '引导用户' else '严格服从指令'}，"
            f"角色描述为“{self.detail}”。"
        )

# 3. FewShotExampleBlock
class FewShotExampleBlock(PromptBlock):
    def __init__(self):
        self.num_options = [0, 1, 2, 3, 4, 5]
        self.order_options = ["随机排序", "按难度递增", "按语义相似度排序"]
        self.content_options = ["展示完整问答", "只展示回答部分"]
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
        return f"该示例部分提供 {self.num} 个样例，样例{self.order}，内容为：{self.content}。"


# 4. ConstraintBlock
class ConstraintBlock(PromptBlock):
    def __init__(self):
        self.num_options = [0, 1, 2, 3, 4, 5]
        self.style_options = ["简洁", "详细"]
        self.format_options = ["段落", "列表"]
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
            f"该模块用于设定模型在回答中必须遵守的限制。当前配置为："
            f"{self.num_constraints} 条约束，风格为“{self.style}”，"
            f"采用{self.format_style}形式展示。"
        )

# 5. CautionBlock
class CautionBlock(PromptBlock):
    def __init__(self):
        self.type_options = ["事实准确", "安全风险", "道德约束", "禁止推测", "避免重复"]
        self.style_options = ["温和提醒", "明确警告", "简洁提示"]
        self.position_options = ["任务前", "任务后"]
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
            f"该模块提供“{self.caution_type}”类提醒，风格为“{self.style}”，"
            f"插入位置在提示{self.position}，共提供 {self.count} 条提醒。"
        )

# 6. SummaryClosureBlock
class SummaryClosureBlock(PromptBlock):
    def __init__(self):
        self.include_summary = ["不包含总结", "包含总结"]
        self.include_next_step = ["不包含后续建议", "包含下一步建议"]
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
        return f"结尾部分{self.summary}，并{self.next_step}。"

def get_all_blocks():
    return [
        TaskObjectiveBlock(),
        RoleConstraintBlock(),
        FewShotExampleBlock(),
        ConstraintBlock(),
        CautionBlock(),
        SummaryClosureBlock(),
    ]