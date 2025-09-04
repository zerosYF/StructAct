from model.model import getOptimModel, getEvalModel
from task.base_task import TaskBase
from program.base_action import OptimizeAction
import concurrent.futures

# Preload models
tester_model = getEvalModel()
analyzer_model = getOptimModel()
rewriter_model = getOptimModel()

class FailureDrivenAction(OptimizeAction):
    """Action from PromptAgent"""
    """Evaluate model output → reflect on failure → revise prompt"""
    def __init__(self, task:TaskBase, name="FailureDrivenAction"):
        super().__init__(task, name)
        self.tester_model = tester_model
        self.analyzer_model = analyzer_model
        self.rewriter_model = rewriter_model
        self.max_resample_attempts = 3
    
    def _batch_api_call(self, inputs:list):
        def call(x):
            try:
                return self.tester_model.api_call(x)
            except Exception as e:
                return f"[ERROR] {str(e)}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(call, inputs))
        return results

    def do(self, current_prompt, trajectory_prompts):
        wrong_answers = ""
        attempts = 0

        while not wrong_answers and attempts < self.max_resample_attempts:
            samples = self.task.sample_train_mcts(self.task.config.batch_size)
            inputs = [self.task.extract_tuple(s)[0] for s in samples]
            golds = [self.task.extract_tuple(s)[1] for s in samples]
            final_inputs = [self.task.inject_final_input(current_prompt, inp) for inp in inputs]
            outputs = self._batch_api_call(final_inputs)

            wrong_answers = "\n\n".join([
                f"[Example {i+1}]\n"
                f"Model's Input: {inp}\n"
                f"Model's Output: {out}\n"
                f"Model's Final Anwser: {self.task._normalize_answer(out)}\n"
                f"Expected Gold Anwser: {self.task._normalize_answer(gold)}\n"
                for i, (inp, out, gold) in enumerate(zip(inputs, outputs, golds)) if self.task.get_reward(out, gold) == 0
            ])
            attempts += 1

        if not wrong_answers:
            return current_prompt
        
        evaluation = (
            "I'm writing prompts for a language model designed for a task.\n"
            f"My current prompt:\n {current_prompt}\n\n"
            f"This prompt gets the following examples wrong:\n {wrong_answers}\n\n"
            "For each wrong example, you should carefully analyze why model response wrong answer, why my prompt leads to wrong answer.\n"
            "Provide comprehensive analysis of the common failure modes, pitfalls, or ambiguities in the prompt that may have contributed to these errors.\n"
            "List me all suggest improvements to the prompt to ensure better generalization.\n"
        )
        analysis = self.analyzer_model.api_call(evaluation)

        rewriting_input = (
            "I'm writing prompts for a language model designed for a task.\n"
            f"My current prompt:\n {current_prompt}\n\n"
            f"This prompt gets the following examples wrong:\n {wrong_answers}\n\n"
            f"Some analysis and suggestions for avoid wrong answers:\n {analysis}\n\n"
            f"There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:\n{trajectory_prompts}\n"
            "The new prompt should solve the current prompt's problems."
            "The new prompt should consider the list of prompts and evolve based on the current prompt."
            "Please rewrite the prompt accordingly. Only output the new prompt."
        )
        rewritten_prompt = self.rewriter_model.api_call(rewriting_input)
        super().do(rewritten_prompt, trajectory_prompts)
        return rewritten_prompt

class SuccessDrivenAction(OptimizeAction):
    def __init__(self, task, name="SuccessDrivenAction", max_resample=3):
        super().__init__(task, name)
        self.tester_model = tester_model          
        self.reasoning_model = analyzer_model    
        self.rewriter_model = rewriter_model      
        self.max_resample = max_resample

    def _batch_api_call(self, inputs: list):
        def call(x):
            try:
                return self.tester_model.api_call(x)
            except Exception as e:
                return f"[ERROR] {str(e)}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(call, inputs))
        return results

    def do(self, current_prompt: str, trajectory_prompts) -> str:
        correct_examples_text = ""
        attempts = 0

        while not correct_examples_text and attempts < self.max_resample:
            samples = self.task.sample_train_mcts(self.task.config.batch_size)
            inputs = [self.task.extract_tuple(s)[0] for s in samples]
            golds  = [self.task.extract_tuple(s)[1] for s in samples]

            final_inputs = [self.task.inject_final_input(current_prompt, inp) for inp in inputs]
            outputs = self._batch_api_call(final_inputs)

            correct_blocks = []
            for i, (inp, out, gold) in enumerate(zip(inputs, outputs, golds)):
                is_correct = self.task.get_reward(out, gold) == 1
                if is_correct:
                    correct_blocks.append(
                        f"[Example {len(correct_blocks)+1}]\n"
                        f"Input: {inp}\n"
                        f"Model Output: {out}\n"
                        f"Model Final Answer: {self.task._normalize_answer(out)}\n"
                        f"Gold Answer: {self.task._normalize_answer(gold)}"
                    )

            correct_examples_text = "\n\n".join(correct_blocks)
            attempts += 1

        if not correct_examples_text:
            return current_prompt

        evaluation = (
            "I'm optimizing a prompt for a language model on a specific task.\n"
            f"My current prompt:\n{current_prompt}\n\n"
            f"Here are some successful examples where the model's prediction matches the correct answer:\n{correct_examples_text}\n\n"
            "For each example, analyze why the model succeeded. "
            "Summarize the key reasoning strategies, invariants, decision rules, or intermediate steps "
            "that should be reinforced in the prompt to generalize better."
        )
        analysis = self.reasoning_model.api_call(evaluation)

        rewriting_input = (
            "I'm optimizing a prompt for a language model.\n"
            f"My current prompt:\n{current_prompt}\n\n"
            f"The following strengths and reasoning strategies were identified from successful examples:\n{analysis}\n\n"
            f"There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:\n{trajectory_prompts}\n\n"
            "The new prompt should consider the list of prompts and evolve based on the current prompt."
            "Please rewrite the prompt to incorporate and emphasize these strengths while keeping it concise and clear.\n"
            "Only output the new prompt."
        )
        rewritten_prompt = self.rewriter_model.api_call(rewriting_input)

        super().do(rewritten_prompt, trajectory_prompts)
        return rewritten_prompt

class CohesionImprover(OptimizeAction):
    """Improve the transitions and cohesion between different prompt blocks."""
    def __init__(self, task, name="CohesionImprover"):
        super().__init__(task, name)
        self.rewriter_model = rewriter_model

    def do(self, current_prompt, trajectory_prompts):

        rewriting_input = (
            f"Current prompt:\n{current_prompt}\n\n"
            f"There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:\n{trajectory_prompts}\n"
            "Please improve the linguistic cohesion and transitions. "
            "The new prompts should consider the list of prompts and evolve based on the current prompt."
            "Only output the revised prompt."
        )
        rewritten_prompt = self.rewriter_model.api_call(rewriting_input)
        super().do(rewritten_prompt, trajectory_prompts)
        return rewritten_prompt

    
def define_full_actions(task: TaskBase):
    return [
        FailureDrivenAction(task),
        SuccessDrivenAction(task),
        CohesionImprover(task),
    ]