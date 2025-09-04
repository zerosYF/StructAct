from model.model import getOptimModel, getEvalModel
from task.base_task import TaskBase
from program.base_action import OptimizeAction
from program.sample_pools import PoolSample
import concurrent.futures

# Preload models
tester_model = getEvalModel()
analyzer_model = getOptimModel()
rewriter_model = getOptimModel()

class FailureDrivenAction(OptimizeAction):
    def __init__(self, task: TaskBase, name="FailureDrivenAction"):
        super().__init__(task, name)
        self.tester_model = tester_model
        self.analyzer_model = analyzer_model
        self.rewriter_model = rewriter_model
        self.max_resample_attempts = 3
        self.use_pool = getattr(self.task.config, "use_sample_pool", False)
        self.hard_ratio = 0.7  # hard样本占比

    def _batch_api_call(self, inputs: list):
        def call(x):
            try:
                return self.tester_model.api_call(x)
            except Exception as e:
                return f"[ERROR] {str(e)}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(call, inputs))

    def sample_and_evaluate(self, current_prompt, sample_pool):
        if self.use_pool and sample_pool:
            hard_k = int(self.task.config.batch_size * self.hard_ratio)
            mix_k = self.task.config.batch_size - hard_k
            samples = sample_pool.sample("hard", k=hard_k) + sample_pool.sample("mixed", k=mix_k)
        else:
            raw_samples = self.task.sample_train_mcts(self.task.config.batch_size)
            samples = [PoolSample(r) for r in raw_samples]

        inputs, golds = [], []
        final_inputs = []
        for s in samples:
            inp, gold = self.task.extract_tuple(s.raw)
            inputs.append(inp)
            golds.append(gold)
            final_inputs.append(self.task.inject_final_input(current_prompt, inp))

        outputs = self._batch_api_call(final_inputs)

        if self.use_pool and sample_pool:
            for s, out, gold in zip(samples, outputs, golds):
                reward = 1 if self.task.get_reward(out, gold) else 0
                sample_pool.add_or_update(s, reward)

        return samples, inputs, outputs, golds

    def format_wrong_examples(self, inputs, outputs, golds):
        return "\n\n".join([
            f"[Example {i+1}]\n"
            f"Model's Input: {inp}\n"
            f"Model's Output: {out}\n"
            f"Model's Final Anwser: {self.task._normalize_answer(out)}\n"
            f"Expected Gold Anwser: {self.task._normalize_answer(gold)}\n"
            for i, (inp, out, gold) in enumerate(zip(inputs, outputs, golds))
            if self.task.get_reward(out, gold) == 0
        ])

    def analyze_and_rewrite(self, current_prompt, wrong_examples, trajectory_prompts):
        evaluation = (
            "I'm writing prompts for a language model designed for a task.\n"
            f"My current prompt:\n {current_prompt}\n\n"
            f"This prompt gets the following examples wrong:\n {wrong_examples}\n\n"
            "For each wrong example, you should carefully analyze why model response wrong answer, why my prompt leads to wrong answer.\n"
            "Provide comprehensive analysis of the common failure modes, pitfalls, or ambiguities in the prompt that may have contributed to these errors.\n"
            "List me all suggest improvements to the prompt to ensure better generalization.\n"
        )
        analysis = self.analyzer_model.api_call(evaluation)

        rewriting_input = (
            "I'm writing prompts for a language model designed for a task.\n"
            f"My current prompt:\n {current_prompt}\n\n"
            f"This prompt gets the following examples wrong:\n {wrong_examples}\n\n"
            f"Some analysis and suggestions for avoid wrong answers:\n {analysis}\n\n"
            f"There are a list of former prompts including the current prompt, and each prompt is modified from its former prompts:\n{trajectory_prompts}\n"
            "The new prompt should solve the current prompt's problems."
            "The new prompt should consider the list of prompts and evolve based on the current prompt."
            "Please rewrite the prompt accordingly. Only output the new prompt."
        )
        return self.rewriter_model.api_call(rewriting_input)

    def do(self, current_prompt, trajectory_prompts, sample_pool=None):
        attempts = 0
        wrong_examples = ""
        while not wrong_examples and attempts < self.max_resample_attempts:
            _, inputs, outputs, golds = self.sample_and_evaluate(current_prompt, sample_pool)
            wrong_examples = self.format_wrong_examples(inputs, outputs, golds)
            attempts += 1

        if not wrong_examples:
            return current_prompt

        rewritten_prompt = self.analyze_and_rewrite(current_prompt, wrong_examples, trajectory_prompts)
        super().do(rewritten_prompt, trajectory_prompts, sample_pool=sample_pool)
        return rewritten_prompt


class SuccessDrivenAction(OptimizeAction):
    def __init__(self, task, name="SuccessDrivenAction", max_resample=3):
        super().__init__(task, name)
        self.tester_model = tester_model
        self.reasoning_model = analyzer_model
        self.rewriter_model = rewriter_model
        self.max_resample = max_resample
        self.use_pool = getattr(self.task.config, "use_sample_pool", False)
        self.success_ratio = 0.7  # success样本占比

    def _batch_api_call(self, inputs: list):
        def call(x):
            try:
                return self.tester_model.api_call(x)
            except Exception as e:
                return f"[ERROR] {str(e)}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(call, inputs))

    def sample_and_evaluate(self, current_prompt, sample_pool):
        if self.use_pool and sample_pool:
            success_k = int(self.task.config.batch_size * self.success_ratio)
            mix_k = self.task.config.batch_size - success_k
            samples = sample_pool.sample("success", k=success_k) + sample_pool.sample("mixed", k=mix_k)
        else:
            raw_samples = self.task.sample_train_mcts(self.task.config.batch_size)
            samples = [PoolSample(r) for r in raw_samples]

        inputs, golds = [], []
        final_inputs = []
        for s in samples:
            inp, gold = self.task.extract_tuple(s.raw)
            inputs.append(inp)
            golds.append(gold)
            final_inputs.append(self.task.inject_final_input(current_prompt, inp))

        outputs = self._batch_api_call(final_inputs)

        if self.use_pool and sample_pool:
            for s, out, gold in zip(samples, outputs, golds):
                reward = 1 if self.task.get_reward(out, gold) else 0
                sample_pool.add_or_update(s, reward)

        return samples, inputs, outputs, golds

    def format_success_examples(self, inputs, outputs, golds):
        correct_blocks = []
        for i, (inp, out, gold) in enumerate(zip(inputs, outputs, golds)):
            if self.task.get_reward(out, gold) == 1:
                correct_blocks.append(
                    f"[Example {len(correct_blocks)+1}]\n"
                    f"Input: {inp}\n"
                    f"Model Output: {out}\n"
                    f"Model Final Answer: {self.task._normalize_answer(out)}\n"
                    f"Gold Answer: {self.task._normalize_answer(gold)}"
                )
        return "\n\n".join(correct_blocks)

    def analyze_and_rewrite(self, current_prompt, success_examples, trajectory_prompts):
        evaluation = (
            "I'm optimizing a prompt for a language model on a specific task.\n"
            f"My current prompt:\n{current_prompt}\n\n"
            f"Here are some successful examples where the model's prediction matches the correct answer:\n{success_examples}\n\n"
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
        return self.rewriter_model.api_call(rewriting_input)

    def do(self, current_prompt, trajectory_prompts, sample_pool=None):
        attempts = 0
        success_examples_text = ""
        while not success_examples_text and attempts < self.max_resample:
            _, inputs, outputs, golds = self.sample_and_evaluate(current_prompt, sample_pool)
            success_examples_text = self.format_success_examples(inputs, outputs, golds)
            attempts += 1

        if not success_examples_text:
            return current_prompt

        rewritten_prompt = self.analyze_and_rewrite(current_prompt, success_examples_text, trajectory_prompts)
        super().do(rewritten_prompt, trajectory_prompts, sample_pool=sample_pool)
        return rewritten_prompt

    
def define_full_actions(task: TaskBase):
    return [
        FailureDrivenAction(task),
        SuccessDrivenAction(task),
    ]