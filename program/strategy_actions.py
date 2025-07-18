from abc import ABC, abstractmethod
from model.model import Model, model
from task.base_task import TaskBase
from program.base_action import OptimizeAction

class TestReflectRewriteAction(OptimizeAction):
    def __init__(self, task, name="TestReflectRewriteAction", original_prompt=None):
        super().__init__(task, name, original_prompt)
        self.tester_model:Model = model
        self.rewriter_model:Model = model
        # 测试模型专用 system_prompt
        self.tester_system_prompt = """
        你是一个严谨的回答生成模型，负责根据输入提示词和问题生成准确且合适的回答。
        请务必保持回答的专业性和准确性。
        """

        # 重写模型专用 system_prompt
        self.rewriter_system_prompt = """
        你是一个提示词优化专家，负责根据当前提示词、模型输出和结构约束，分析并优化提示词。
        请返回一个经过改进、更加清晰、有效且符合约束的提示词文本。
        """

    def do(self, current_prompt, template_description):
        # Step 1: 采样与构造输入
        samples = self.task.sample_train()
        inputs = [self.task.extract_tuple(s)[0] for s in samples]
        gold_answers = [self.task.extract_tuple(s)[1] for s in samples]
        final_inputs = [self.task.inject_final_input(current_prompt, inp) for inp in inputs]

        # Step 2: 测试模型生成回答
        answers = [self.tester_model.api_call(self.tester_system_prompt, final_inp) for final_inp in final_inputs]

        # Step 3: 构造统一反馈内容块
        evaluation_blocks = []
        for i, (inp, ans, gold) in enumerate(zip(inputs, answers, gold_answers)):
            block = (
                f"【示例 {i+1}】\n"
                f"输入：{inp}\n"
                f"模型回答：{ans}\n"
                f"参考答案：{gold}\n"
            )
            evaluation_blocks.append(block)
        combined_eval_input = "\n".join(evaluation_blocks)

        # Step 4: 构造整体评估-改写 prompt（直接传入 rewriter）
        rewriting_prompt = (
            f"当前提示词：\n{current_prompt}\n\n"
            f"模型输出示例：\n{combined_eval_input}\n\n"
            f"提示词结构约束：\n{template_description}\n\n"
            f"请根据以上内容，综合判断当前提示词的问题并优化它，返回修改后的提示词："
        )

        # Step 5: 模型生成优化提示词
        rewritten_prompt = self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_prompt)

        return rewritten_prompt

class ConstraintBlockRefiner(OptimizeAction):
    def __init__(self, task, name = "ConstraintBlockRefiner", original_prompt = None):
        super().__init__(task, name, original_prompt)
        self.model:Model = model

    def do(self, current_prompt, template_description):
        samples = self.task.sample_train()
        violations = []

        for s in samples:
            input_, expected = self.task.extract_tuple(s)
            content = (
            f"提示词：\n{current_prompt}\n\n"
            f"模型输入：\n{input_}\n\n"
            f"模型输出：\n{expected}\n\n"
            "请指出是否有违反提示词约束的地方，若无则返回空字符串。"
            )
            response = self.model.api_call(self.system_prompt, content)
            if response:
                violations.append(response)

        constraint_summary = "\n".join(set(violations))

        system_prompt = (
            "你是一个提示词工程师，负责根据模型输出的约束反馈优化提示词。\n"
            "请根据以下内容，结合约束模块的文本，改写当前提示词，使其更符合约束要求。\n"
            "只需返回改写后的完整提示词。"
        )

        content = (
            f"当前提示词：\n{current_prompt}\n\n"
            f"模型约束反馈：\n{constraint_summary}\n\n"
            f"约束模块文本：\n{template_description}\n\n"
            f"请改写提示词："
        )

        rewritten_prompt = self.rewriter_model.api_call(system_prompt, content)

        return rewritten_prompt

class FewShotExampleBuilder(OptimizeAction):
    def __init__(self, task, name="FewShotExampleBuilder", original_prompt=None):
        super().__init__(task, name, original_prompt)
        self.builder_model: Model = model
        self.rewriter_model: Model = model
        self.builder_system_prompt = """
        你是一个提示词工程助手。你将根据任务提示词结构描述、已有的示例问答对，以及当前提示词内容，构造出新的 few-shot 问答对。
        你的目标是补充出 1~2 个高质量的问答对，这些问答对应与已有示例风格一致、难度适中，能更好地引导语言模型完成任务。只输出问答对文本，不加解释。
        """
        self.rewriter_system_prompt = """
        你是一个提示词优化专家。你需要将当前提示词重新组织，使其结构更加清晰、样例合理融入，并保持整体逻辑一致。
        你将使用给定的 few-shot 示例、结构描述与当前提示词，生成优化后的新提示词。
        """

    def do(self, current_prompt, template_description):
        samples = self.task.sample_train()
        original_fewshot = samples[:2]

        original_text = "\n\n".join([f"Q: {s['input']}\nA: {s['output']}" for s in original_fewshot])

        builder_prompt = (
            f"当前任务提示词结构说明：\n{template_description}\n\n"
            f"当前提示词内容：\n{current_prompt}\n\n"
            f"已有的真实问答对示例：\n{original_text}\n\n"
            f"请基于当前提示词和以上示例，构造1~2个新的few-shot问答对示例文本，"
            f"以帮助模型更好理解任务要求。"
        )
        new_fewshot_text = self.builder_model.api_call(self.builder_system_prompt, builder_prompt)

        rewriting_prompt = (
            f"当前提示词：\n{current_prompt}\n\n"
            f"可选的新 few-shot 示例如下：\n{new_fewshot_text}\n\n"
            f"提示结构描述如下：\n{template_description}\n\n"
            f"请将新示例合理融合进提示词中，保持整体结构清晰自然，输出最终重写后的提示词："
        )

        rewritten_prompt = self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_prompt)
        return rewritten_prompt

class StructureOptimizerByPerformance(OptimizeAction):
    def __init__(self, task, name="StructureOptimizerByPerformance", original_prompt=None):
        super().__init__(task, name, original_prompt)
        self.tester_model:Model = model
        self.rewriter_model:Model = model
        self.tester_system_prompt = """
        你是一个严谨的回答生成模型，负责根据输入提示词和问题生成准确且合适的回答。
        """
        self.rewriter_system_prompt = """
        你是一个提示词优化专家，负责根据当前提示词、模型表现反馈和结构约束，优化提示词表达，使其更明确有效。
        """

    def do(self, current_prompt, template_structure):
        samples = self.task.sample_train()

        inputs = [self.task.extract_tuple(s)[0] for s in samples]
        golds = [self.task.extract_tuple(s)[1] for s in samples]
        final_inputs = [self.task.inject_final_input(current_prompt, inp) for inp in inputs]

        answers = [self.tester_model.api_call(self.tester_system_prompt, fin) for fin in final_inputs]
        scores = [self.evaluator_model.score(ans, gold) for ans, gold in zip(answers, golds)]
        avg_score = sum(scores) / len(scores)

        feedback = f"当前提示词在测试样本上的平均完成度为 {avg_score:.2f}。请基于该反馈改进提示词，使其更有效。"

        rewriting_content = (
            f"当前提示词：\n{current_prompt}\n\n"
            f"任务完成度反馈：\n{feedback}\n\n"
            f"提示词结构约束：\n{template_structure}\n\n"
            "请输出改进后的完整提示词："
        )
        rewritten_prompt = self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_content)
        return rewritten_prompt

class InstructionSimplifierByAbstraction(OptimizeAction):
    def __init__(self, task, name="InstructionSimplifierByAbstraction", original_prompt=None):
        super().__init__(task, name, original_prompt)
        self.evaluator_model: Model = model
        self.evaluator_system_prompt = """
        你是一个具有概况总结能力的论证专家，能够根据多组问答对总结这一类任务的任务目标
        """
        self.rewriter_model:Model = model
        self.rewriter_system_prompt = """
        你是一个提示词优化专家，负责抽象总结任务目标并将其融入提示词，使提示更简洁且表达明确。
        """

    def do(self, current_prompt, template_structure):
        samples = self.task.sample_train()
        abstract_goal = self.evaluator_model.api_call(
            self.evaluator_system_prompt,
            f"请基于以下问答对进行抽象总结：\n" + "\n" + self.task.samples2text(samples) + "\n\n并请给出简洁明确的总结。"
        )

        rewriting_content = (
            f"当前提示词：\n{current_prompt}\n\n"
            f"抽象任务目标总结：\n{abstract_goal}\n\n"
            f"提示词结构约束：\n{template_structure}\n\n"
            "请基于以上内容，优化提示词表达，输出完整改写的提示词："
        )
        rewritten_prompt = self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_content)
        return rewritten_prompt

# 工厂函数
def define_full_actions(task: TaskBase):
    return [
        TestReflectRewriteAction(task),
        ConstraintBlockRefiner(task),
        FewShotExampleBuilder(task),
        StructureOptimizerByPerformance(task),
        InstructionSimplifierByAbstraction(task),
    ]