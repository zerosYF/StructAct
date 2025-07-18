from abc import ABC, abstractmethod
from model.model import Model, getModel
from task.base_task import TaskBase
class OptimizeAction(ABC):
    def __init__(self, 
                 task:TaskBase,
                 name:str=None, 
                 original_prompt:str=None,
                 ):
        self.name = name
        self.system_prompt = f"""
            你是一个提示词工程师,你的任务是针对当前提示词给出优化后的提示词。
            只给出修改过后的提示词即可，并保持中性、务实的描述风格。
            未经过任何修改的初始提示词是:{original_prompt}。
            """
        self.task = task
    
    @abstractmethod
    def do(self, 
           current_prompt: str, 
           template_description: str) -> str:
        """
        samples: List[dict] like [{"input": ..., "output": ...}, ...]
        structure: 当前的prompt结构模板
        return: 更新后的提示词
        """
        pass

class StructureSyncAction(OptimizeAction):
    def __init__(self, task, name="StructureSyncAction", original_prompt=None):
        super().__init__(task, name, original_prompt)
        self.rewriter_model: Model = getModel()

    def do(self, current_prompt, template_description):
        # 采样样本，用于填充具体内容（如 fewshot、约束等）
        samples = self.task.sample_train()

        # 构造 few-shot 示例片段（抽取真实问答）
        example_texts = self.task.samples2text(samples)

        system_prompt = (
            "你是一个提示词构造助手，你的任务是根据结构模板与样本信息，重写提示词，使其内容与结构一致。\n"
            "不允许更改结构参数，只更新提示内容。\n"
            "请保持风格统一、表达准确，避免冗余。"
        )

        rewrite_prompt = (
            f"【当前提示词】：\n{current_prompt}\n\n"
            f"【结构模板约束说明】：\n{template_description}\n\n"
            f"【参考样本问答对】：\n{example_texts}\n\n"
            f"请基于当前结构，利用样本中的信息，补充/重写提示内容，使之与结构一致。\n"
            f"禁止改动结构设置。"
        )

        rewritten_prompt = self.rewriter_model.api_call(system_prompt, rewrite_prompt)
        return rewritten_prompt

    