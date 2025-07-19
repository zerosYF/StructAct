from abc import ABC, abstractmethod
from model.model import Model, getModel
from task.base_task import TaskBase
from program.base_action import OptimizeAction

# Preload models
tester_model = getModel()
rewriter_model = getModel()
checker_model = getModel()
builder_model = getModel()
evaluator_model = getModel()

class TestReflectRewriteAction(OptimizeAction):
    """Evaluate model output → reflect on failure → revise prompt"""
    def __init__(self, task, name="TestReflectRewriteAction"):
        super().__init__(task, name)
        self.tester_model = tester_model
        self.rewriter_model = rewriter_model
        self.tester_system_prompt = "You're a QA assistant. Given prompt and input, generate accurate answers."
        self.rewriter_system_prompt = "You're a prompt engineer. Refine the prompt based on model outputs and structure."

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        samples = self.task.sample_train()
        inputs = [self.task.extract_tuple(s)[0] for s in samples]
        golds = [self.task.extract_tuple(s)[1] for s in samples]
        final_inputs = [self.task.inject_final_input(current_prompt, inp) for inp in inputs]
        outputs = [self.tester_model.api_call(self.tester_system_prompt, x) for x in final_inputs]

        evaluation = "\n".join([
            f"[Example {i+1}]\nInput: {inp}\nOutput: {out}\nExpected: {gold}"
            for i, (inp, out, gold) in enumerate(zip(inputs, outputs, golds))
        ])
        prompt = (
            f"Current prompt:\n{current_prompt}\n\n"
            f"Model output examples:\n{evaluation}\n\n"
            f"Prompt structure:\n{template_description}\n\n"
            "Please revise the prompt accordingly:"
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, prompt)

class ConstraintBlockRefiner(OptimizeAction):
    """Check constraint violations and fix prompt"""
    def __init__(self, task, name="ConstraintBlockRefiner"):
        super().__init__(task, name)
        self.checker_model = checker_model
        self.rewriter_model = rewriter_model
        self.checker_system_prompt = "You're a constraint checker. Report any violations in model output."
        self.rewriter_system_prompt = "You're a prompt engineer. Refine prompt to better enforce constraints."

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        samples = self.task.sample_train()
        violations = []

        for s in samples:
            inp, expected = self.task.extract_tuple(s)
            check_input = f"Prompt:\n{current_prompt}\n\nInput: {inp}\nExpected: {expected}"
            response = self.checker_model.api_call(self.checker_system_prompt, check_input)
            if response.strip():
                violations.append(response)

        feedback = "\n".join(set(violations))
        prompt = (
            f"Current prompt:\n{current_prompt}\n\n"
            f"Constraint feedback:\n{feedback}\n\n"
            f"Constraint block:\n{template_description}\n\n"
            "Please revise the prompt accordingly:"
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, prompt)

class FewShotExampleBuilder(OptimizeAction):
    """Add new few-shot examples and integrate into prompt"""
    def __init__(self, task, name="FewShotExampleBuilder"):
        super().__init__(task, name)
        self.builder_model = builder_model
        self.rewriter_model = rewriter_model
        self.builder_system_prompt = "Generate 1–2 new few-shot QA pairs based on structure and current prompt."
        self.rewriter_system_prompt = "Integrate new examples into the prompt, keeping structure consistent."

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        original_examples = self.task.sample_train()
        original_text = self.task.samples2text(original_examples)

        builder_input = (
            f"Prompt structure:\n{template_description}\n\n"
            f"Current prompt:\n{current_prompt}\n\n"
            f"Existing examples:\n{original_text}\n\n"
            "Add 1–2 new high-quality QA examples."
        )
        new_examples = self.builder_model.api_call(self.builder_system_prompt, builder_input)

        rewriting_input = (
            f"Prompt:\n{current_prompt}\n\n"
            f"New examples:\n{new_examples}\n\n"
            f"Structure description:\n{template_description}\n\n"
            "Integrate examples and return updated prompt."
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_input)

class OptimizerByPerformance(OptimizeAction):
    """Trace reasoning failures and improve prompt"""
    def __init__(self, task, name="OptimizerByPerformance"):
        super().__init__(task, name)
        self.tester_model = tester_model
        self.rewriter_model = rewriter_model
        self.tester_system_prompt = "Given input and prompt, try to reason out expected output."
        self.rewriter_system_prompt = "Refine the prompt to improve reasoning and effectiveness."

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        samples = self.task.sample_train()
        inputs = [self.task.extract_tuple(s)[0] for s in samples]
        golds = [self.task.extract_tuple(s)[1] for s in samples]
        final_inputs = [self.task.inject_final_input(current_prompt, inp) for inp in inputs]
        responses = [self.tester_model.api_call(self.tester_system_prompt, x) for x in final_inputs]

        analysis = "\n".join([
            f"Input: {inp}\nGold: {gold}\nModel reasoning: {resp}\n"
            for inp, gold, resp in zip(inputs, golds, responses)
        ])
        rewriting_prompt = (
            f"Current prompt:\n{current_prompt}\n\n"
            f"Analysis of performance:\n{analysis}\n\n"
            f"Prompt structure:\n{template_description}\n\n"
            "Please optimize the prompt accordingly:"
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_prompt)

class InstructionSimplifierByAbstraction(OptimizeAction):
    """Abstract task intent from examples and inject into prompt"""
    def __init__(self, task, name="InstructionSimplifierByAbstraction"):
        super().__init__(task, name)
        self.evaluator_model = evaluator_model
        self.rewriter_model = rewriter_model
        self.evaluator_system_prompt = "Summarize task goal based on QA pairs."
        self.rewriter_system_prompt = "Inject abstract task intent into prompt for clarity and conciseness."

    def do(self, current_prompt, template_structure):
        super().do(current_prompt, template_structure)
        samples = self.task.sample_train()
        qa_text = self.task.samples2text(samples)
        summary_input = f"{self.evaluator_system_prompt}\nQA pairs:\n{qa_text}"
        abstract_goal = self.evaluator_model.api_call(self.evaluator_system_prompt, summary_input)

        rewriting_input = (
            f"Prompt:\n{current_prompt}\n\n"
            f"Abstracted task goal:\n{abstract_goal}\n\n"
            f"Prompt structure:\n{template_structure}\n\n"
            "Please revise the prompt accordingly:"
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_input)

def define_full_actions(task: TaskBase):
    return [
        TestReflectRewriteAction(task),
        ConstraintBlockRefiner(task),
        FewShotExampleBuilder(task),
        OptimizerByPerformance(task),
        InstructionSimplifierByAbstraction(task),
    ]