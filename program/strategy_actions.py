from abc import ABC, abstractmethod
from model.model import Model, getModel
from task.base_task import TaskBase
from program.base_action import OptimizeAction

tester_model = getModel()
rewriter_model = getModel()
checker_model = getModel()
builder_model = getModel()
evaluator_model = getModel()

class TestReflectRewriteAction(OptimizeAction):
    def __init__(self, task, name="TestReflectRewriteAction"):
        super().__init__(task, name)
        self.tester_model: Model = tester_model
        self.rewriter_model: Model = rewriter_model

        self.tester_system_prompt = """
        你是一个严谨的回答生成模型，负责根据输入提示词和问题生成准确且合适的回答。
        请务必保持回答的专业性和准确性。
        """
        self.rewriter_system_prompt = """
        你是一个提示词优化专家，负责根据当前提示词、模型输出和结构约束，分析并优化提示词。
        请返回一个经过改进、更加清晰、有效且符合约束的提示词文本。
        """

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        samples = self.task.sample_train()
        inputs = [self.task.extract_tuple(s)[0] for s in samples]
        gold_answers = [self.task.extract_tuple(s)[1] for s in samples]
        final_inputs = [self.task.inject_final_input(current_prompt, inp) for inp in inputs]

        answers = [self.tester_model.api_call(self.tester_system_prompt, fin) for fin in final_inputs]

        evaluation_blocks = []
        for i, (inp, ans, gold) in enumerate(zip(inputs, answers, gold_answers)):
            evaluation_blocks.append(
                f"【示例 {i+1}】\n输入：{inp}\n模型回答：{ans}\n参考答案：{gold}\n"
            )
        combined_eval_input = "\n".join(evaluation_blocks)

        rewriting_prompt = (
            f"当前提示词：\n{current_prompt}\n\n"
            f"模型输出示例：\n{combined_eval_input}\n\n"
            f"提示词结构约束：\n{template_description}\n\n"
            f"请根据以上内容，综合判断当前提示词的问题并优化它，返回修改后的提示词："
        )

        return self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_prompt)

class ConstraintBlockRefiner(OptimizeAction):
    def __init__(self, task, name="ConstraintBlockRefiner"):
        super().__init__(task, name)
        self.checker_model: Model = checker_model
        self.rewriter_model: Model = rewriter_model

        self.checker_system_prompt = """
        你是一个提示词约束检查员，负责判断输出是否违反提示词设定的限制或禁忌。若存在，请返回违反说明；若无，请返回空字符串。
        """
        self.rewriter_system_prompt = """
        你是一个提示词工程师，负责根据模型输出的约束反馈优化提示词。
        请根据反馈内容，结合结构描述中的约束部分改写提示词，使其更符合要求。
        """

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        samples = self.task.sample_train()
        violations = []

        for s in samples:
            input_, expected = self.task.extract_tuple(s)
            checker_input = (
                f"提示词：\n{current_prompt}\n\n"
                f"模型输入：{input_}\n期望输出：{expected}"
            )
            response = self.checker_model.api_call(self.checker_system_prompt, checker_input)
            if response.strip():
                violations.append(response)

        constraint_summary = "\n".join(set(violations))
        rewriting_input = (
            f"当前提示词：\n{current_prompt}\n\n"
            f"模型约束反馈：\n{constraint_summary}\n\n"
            f"约束模块文本：\n{template_description}\n\n"
            "请改写提示词："
        )

        return self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_input)

class FewShotExampleBuilder(OptimizeAction):
    def __init__(self, task, name="FewShotExampleBuilder"):
        super().__init__(task, name)
        self.builder_model: Model = builder_model
        self.rewriter_model: Model = rewriter_model

        self.builder_system_prompt = """
        你是一个提示词构造助手。根据已有的few-shot示例、当前提示词和结构说明，构造1~2个新的高质量问答对。
        """
        self.rewriter_system_prompt = """
        你是一个提示词优化专家。请将提供的新few-shot示例自然融合进提示词中，保持结构清晰和一致性。
        """

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        samples = self.task.sample_train()
        original_fewshot = samples

        original_text = self.task.samples2text(original_fewshot)

        builder_prompt = (
            f"当前提示词结构说明：\n{template_description}\n\n"
            f"当前提示词：\n{current_prompt}\n\n"
            f"已有问答示例：\n{original_text}\n\n"
            f"请补充1~2个风格一致的新问答示例。"
        )

        new_examples = self.builder_model.api_call(self.builder_system_prompt, builder_prompt)

        rewriting_prompt = (
            f"当前提示词：\n{current_prompt}\n\n"
            f"可用的few-shot示例：\n采样示例:\n{original_text}\n新构造示例:\n{new_examples}\n\n"
            f"结构约束说明：\n{template_description}\n\n"
            f"请根据结构约束选择示例自然嵌入提示词并返回新提示词。"
        )

        return self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_prompt)

class OptimizerByPerformance(OptimizeAction):
    def __init__(self, task, name="OptimizerByPerformance"):
        super().__init__(task, name)
        self.tester_model: Model = tester_model
        self.rewriter_model: Model = rewriter_model

        self.tester_system_prompt = """
        你是一个任务执行助手，请根据提示词和输入和对应参考答案，尝试推演输入到输出的分析过程。
        """
        self.rewriter_system_prompt = """
        你是一个提示词优化专家，负责根据任务表现反馈和结构约束，优化提示词表达，使其更明确有效。
        """

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        samples = self.task.sample_train()
        inputs = [self.task.extract_tuple(s)[0] for s in samples]
        golds = [self.task.extract_tuple(s)[1] for s in samples]
        final_inputs = [self.task.inject_final_input(current_prompt, inp) for inp in inputs]

        responses = [self.tester_model.api_call(self.tester_system_prompt, x) for x in final_inputs]

        prompt = "\n".join([
            f"问题输入：{inp}\n模型回答：{gold}\n中间推理过程：{response}\n" for inp, gold, response in zip(inputs, golds, responses)
        ])

        rewriting_prompt = (
            f"当前提示词：\n{current_prompt}\n\n"
            f"对采样问答的分析：{prompt}\n\n"
            f"结构约束如下：\n{template_description}\n\n"
            "请根据上述信息优化提示词表达，使其更有效。"
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_prompt)

class InstructionSimplifierByAbstraction(OptimizeAction):
    def __init__(self, task, name="InstructionSimplifierByAbstraction"):
        super().__init__(task, name)
        self.evaluator_model: Model = evaluator_model
        self.rewriter_model: Model = rewriter_model

        self.evaluator_system_prompt = """
        你是一个任务总结助手，擅长从多个问答对中抽象，总结任务目标等全局信息。
        """
        self.rewriter_system_prompt = """
        你是一个提示词优化专家，负责将抽象总结出的任务目标等全局信息融入提示词，使其更简洁且表达明确。
        """

    def do(self, current_prompt, template_structure):
        super().do(current_prompt, template_structure)
        samples = self.task.sample_train()
        qa_text = self.task.samples2text(samples)

        abstract_prompt = (
            f"{self.evaluator_system_prompt}\n请基于以下问答对总结这类任务目标，只输出总结文本：\n" + qa_text
        )
        abstract_goal = self.evaluator_model.api_call(self.evaluator_system_prompt, abstract_prompt)

        rewriting_content = (
            f"当前提示词：\n{current_prompt}\n\n"
            f"抽象总结的任务目标等全局信息：\n{abstract_goal}\n\n"
            f"结构约束：\n{template_structure}\n\n"
            "请根据抽象总结出的任务目标等全局信息重写提示词。"
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_content)

# 工厂函数
def define_full_actions(task: TaskBase):
    return [
        TestReflectRewriteAction(task),
        ConstraintBlockRefiner(task),
        FewShotExampleBuilder(task),
        OptimizerByPerformance(task),
        InstructionSimplifierByAbstraction(task),
    ]