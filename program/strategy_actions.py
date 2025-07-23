from abc import ABC, abstractmethod
from model.model import Model, getModel
from task.base_task import TaskBase
from program.base_action import OptimizeAction
import concurrent.futures

# Preload models
tester_model = getModel()
analyzer_model = getModel()
rewriter_model = getModel()
checker_model = getModel()
builder_model = getModel()
evaluator_model = getModel()

class TestReflectRewriteAction(OptimizeAction):
    """Evaluate model output → reflect on failure → revise prompt"""
    def __init__(self, task, name="TestReflectRewriteAction"):
        super().__init__(task, name)
        self.tester_model = tester_model
        self.analyzer_model = analyzer_model
        self.rewriter_model = rewriter_model
        self.tester_system_prompt = "You're a QA assistant. Given prompt and input, generate accurate answers."
        self.analyzer_system_prompt = "Analysis the model outputs against expected answers. Provide detailed feedback."
        self.rewriter_system_prompt = "You're a prompt engineer. Refine the prompt based on model outputs and structure."
    
    def _batch_api_call(self, inputs:list):
        def call(x):
            try:
                return self.tester_model.api_call(self.tester_system_prompt, x)
            except Exception as e:
                return f"[ERROR] {str(e)}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(call, inputs))
        return results

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        samples = self.task.sample_train()
        inputs = [self.task.extract_tuple(s)[0] for s in samples]
        golds = [self.task.extract_tuple(s)[1] for s in samples]
        final_inputs = [self.task.inject_final_input(current_prompt, inp) for inp in inputs]
        outputs = self._batch_api_call(final_inputs)

        evaluation = "\n".join([
            f"[Example {i+1}]\nInput: {inp}\nOutput: {out}\nExpected: {gold}"
            for i, (inp, out, gold) in enumerate(zip(inputs, outputs, golds))
        ])
        evaluation = (
            "You should analyze why the model outputs the right or wrong answer, "
            "and suggest improvements to the prompt to ensure better generalization.\n"
            + evaluation
        )
        analysis = self.analyzer_model.api_call(self.analyzer_system_prompt, evaluation)

        prompt = (
            f"Current prompt:\n{current_prompt}\n"
            f"Analysis:\n{analysis}\n"
            f"Prompt structure:\n{template_description}\n"
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
        new_examples += "\n" + original_text  # 当前是字符串拼接，未去重  # Combine new examples with existing ones

        rewriting_input = (
            f"Prompt:\n{current_prompt}\n"
            f"New examples model created you can select:\n{new_examples}\n"
            f"Structure description:\n{template_description}\n"
            f"Integrate examples and return updated prompt.\n" 
            f"You can replace existing examples or add new ones.\n"
            f"You should select the best examples and keep the structure consistent.\n\n"
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_input)

class InstructionSimplifierByAbstraction(OptimizeAction):
    """Abstract task intent from examples and inject into prompt"""
    def __init__(self, task, name="InstructionSimplifierByAbstraction"):
        super().__init__(task, name)
        self.evaluator_model = evaluator_model
        self.rewriter_model = rewriter_model
        self.evaluator_system_prompt = "Summarize task goal based on QA pairs."
        self.rewriter_system_prompt = "Inject abstract task intent into prompt for clarity and conciseness."

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        samples = self.task.sample_train()
        qa_text = self.task.samples2text(samples)
        summary_input = f"QA pairs:\n{qa_text}"
        abstract_goal = self.evaluator_model.api_call(self.evaluator_system_prompt, summary_input)

        rewriting_input = (
            f"Prompt:\n{current_prompt}\n"
            f"Abstracted task goal:\n{abstract_goal}\n"
            f"Prompt structure:\n{template_description}\n"
            "Please revise the prompt accordingly:"
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_input)

class LexicalSimplifier(OptimizeAction):
    """Simplify wording to improve clarity and readability without changing structure."""
    def __init__(self, task, name="LexicalSimplifier"):
        super().__init__(task, name)
        self.rewriter_model = rewriter_model
        self.rewriter_system_prompt = (
            "You are a prompt editor. Simplify the wording of the prompt to make it more readable, "
            "without changing its structure or meaning."
        )

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        prompt = (
            f"Prompt:\n{current_prompt}\n\n"
            f"Prompt structure:\n{template_description}\n\n"
            f"Revise the prompt by simplifying complex phrases, eliminating redundancy, and improving clarity."
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, prompt)

class StyleHarmonizer(OptimizeAction):
    """Align the style and tone of all blocks in the prompt."""
    def __init__(self, task, name="StyleHarmonizer"):
        super().__init__(task, name)
        self.rewriter_model = rewriter_model
        self.rewriter_system_prompt = (
            "You are a prompt stylist. Adjust the wording to ensure consistent tone and style across the prompt, "
            "based on the structure."
        )

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        prompt = (
            f"Prompt:\n{current_prompt}\n\n"
            f"Structure:\n{template_description}\n\n"
            f"Please make sure all parts follow the same tone and are stylistically consistent (e.g., formal or instructional)."
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, prompt)

class CohesionImprover(OptimizeAction):
    """Improve the transitions and cohesion between different prompt blocks."""
    def __init__(self, task, name="CohesionImprover"):
        super().__init__(task, name)
        self.rewriter_model = rewriter_model
        self.rewriter_system_prompt = (
            "You're a prompt cohesion expert. Improve the transitions between sections so that the overall prompt flows naturally."
        )

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        prompt = (
            f"Prompt:\n{current_prompt}\n\n"
            f"Prompt structure:\n{template_description}\n\n"
            f"Please revise the transitions and phrasing between blocks to improve fluency and coherence."
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, prompt)

class AmbiguityReducer(OptimizeAction):
    """Identify and remove ambiguous phrases in the prompt to reduce misunderstanding."""
    def __init__(self, task, name="AmbiguityReducer"):
        super().__init__(task, name)
        self.rewriter_model = rewriter_model
        self.rewriter_system_prompt = (
            "You are an ambiguity checker. Rewrite the prompt to eliminate vague, ambiguous, or underspecified parts."
        )

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        prompt = (
            f"Prompt:\n{current_prompt}\n\n"
            f"Prompt structure:\n{template_description}\n\n"
            f"Please revise any parts that might be unclear or ambiguous, ensuring the prompt is fully precise."
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, prompt)
    
def define_full_actions(task: TaskBase):
    return [
        TestReflectRewriteAction(task),
        FewShotExampleBuilder(task),
        InstructionSimplifierByAbstraction(task),
        LexicalSimplifier(task),
        StyleHarmonizer(task),
        CohesionImprover(task),
        AmbiguityReducer(task),
    ]