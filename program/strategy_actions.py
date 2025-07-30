from model.model import Model, getOptimModel
from task.base_task import TaskBase
from program.base_action import OptimizeAction
import concurrent.futures

# Preload models
tester_model = getOptimModel()
analyzer_model = getOptimModel()
rewriter_model = getOptimModel()
reasoning_model = getOptimModel()

class TestReflectRewriteAction(OptimizeAction):
    """Action from PromptAgent"""
    """Evaluate model output → reflect on failure → revise prompt"""
    def __init__(self, task:TaskBase, name="TestReflectRewriteAction"):
        super().__init__(task, name)
        self.tester_model = tester_model
        self.analyzer_model = analyzer_model
        self.rewriter_model = rewriter_model
    
    def _batch_api_call(self, inputs:list):
        def call(x):
            try:
                return self.tester_model.api_call(x)
            except Exception as e:
                return f"[ERROR] {str(e)}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(call, inputs))
        return results

    def do(self, current_prompt, template_description):
        samples = self.task.sample_train_mcts(self.task.config.batch_size)
        inputs = [self.task.extract_tuple(s)[0] for s in samples]
        golds = [self.task.extract_tuple(s)[1] for s in samples]
        final_inputs = [self.task.inject_final_input(current_prompt, inp) for inp in inputs]
        outputs = self._batch_api_call(final_inputs)

        wrong_answers = "\n\n".join([
            f"[Example {i+1}]\n"
            f"Model's Input: {inp}\n"
            f"Model's Output: {out}\n"
            f"Model's Final Anwser: {self.task._normalize_answer(out)}"
            f"Expected Gold Anwser: {self.task._normalize_answer(gold)}"
            for i, (inp, out, gold) in enumerate(zip(inputs, outputs, golds)) if self.task.get_reward(out, gold) == 0
        ])
        evaluation = (
            "I'm writing prompts for a language model designed for a task.\n"
            f"My current prompt:\n {current_prompt}"
            f"This prompt gets the following examples wrong:\n {wrong_answers}"
            "For every example, you should carefully analyze why model response wrong answer, why my prompt leads to wrong answer."
            "List me all suggest improvements to the prompt to ensure better generalization.\n"
        )
        analysis = self.analyzer_model.api_call(evaluation)

        rewriting_input = (
            "I'm writing prompts for a language model designed for a task.\n"
            f"My current prompt:\n {current_prompt}\n\n"
            f"But this prompt gets the following examples wrong:\n {wrong_answers}\n\n"
            f"Some suggestions for avoid wrong answers:\n {analysis}\n\n"
            f"My prompt template description:\n{template_description}\n\n"
            f"Based on the above information, please rewrite new prompts:\n"
            """
            - Your revision must strictly follow my prompt template description.
            - The new prompts should solve the current prompt's problems.
            - Only output the revised prompt **without any explanation**.
            - Do not change the prompt structure layout or add new blocks, your new information should be in ​appropriate​ block.
            """
        )
        rewritten_prompt = self.rewriter_model.api_call(rewriting_input)
        super().do(rewritten_prompt, template_description)
        return rewritten_prompt

class FewShotExampleLearning(OptimizeAction):
    """Add new few-shot examples and integrate into prompt"""
    def __init__(self, task, name="FewShotExampleLearning"):
        super().__init__(task, name)
        self.reasoning_model = reasoning_model
        self.rewriter_model = rewriter_model

    def do(self, current_prompt, template_description):
        original_examples = self.task.sample_train_mcts(self.task.config.batch_size)
        some_examples = self.task.samples2text(original_examples)

        evaluation = (
            "I'm writing prompts for a language model designed for a task.\n"
            f"My current prompt:\n {current_prompt}\n"
            f"There are some examples in current task:\n {some_examples}\n"
            "For every example, you should carefully analyze and reason how to get output from input.\n"
            "List me all suggest improvements to the prompt to ensure better generalization.\n"
        )
        analysis = self.reasoning_model.api_call(evaluation)

        rewriting_input = (
            "I'm writing prompts for a language model designed for a task.\n"
            f"My current prompt:\n{current_prompt}\n"
            f"There are some suggestions to improve current prompt:\n{analysis}\n"
            f"My current prompt template description:\n{template_description}\n"
            """
            - The revised prompt **must strictly follow** the template description.
            - Do not change the prompt structure layout or add new blocks, your new information should be in ​appropriate​ block.
            - Only output the **final revised prompt**.
            """
        )
        rewritten_prompt = self.rewriter_model.api_call(rewriting_input)
        super().do(rewritten_prompt, template_description)
        return rewritten_prompt

class CohesionImprover(OptimizeAction):
    """Improve the transitions and cohesion between different prompt blocks."""
    def __init__(self, task, name="CohesionImprover"):
        super().__init__(task, name)
        self.rewriter_model = rewriter_model

    def do(self, current_prompt, template_description):

        rewriting_input = (
            "I'm writing prompts for a language model designed for a task.\n"
            f"My current prompt:\n{current_prompt}\n"
            f"My prompt template desciption:\n{template_description}\n"
            """
            - Improve only the linguistic cohesion and transitions between blocks in my prompt.\n
            - Do not change the blocks order, content meaning, or formatting.\n
            - Use prompt template description to ensure the block layout is preserved.\n
            - Keep all blocks distinct and intact.\n
            - Add brief connective cues only where necessary (e.g., intro phrases, bridges).\n
            - Output ONLY the revised prompt — do not add explanations, comments, or any extra content.\n"""
        )
        rewritten_prompt = self.rewriter_model.api_call(rewriting_input)
        super().do(rewritten_prompt, template_description)
        return rewritten_prompt

    
def define_full_actions(task: TaskBase):
    return [
        TestReflectRewriteAction(task),
        FewShotExampleLearning(task),
        CohesionImprover(task),
    ]