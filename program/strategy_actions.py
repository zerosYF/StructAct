from model.model import getOptimModel, getEvalModel
from task.base_task import TaskBase
from program.base_action import OptimizeAction
from program.sample_pools import PoolSample, SampleType, DynamicSamplePool
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
        self.sample_failure_counter = 0  # 连续失败次数

    def _batch_api_call(self, inputs: list):
        def call(x):
            try:
                return self.tester_model.api_call(x)
            except Exception as e:
                return f"[ERROR] {str(e)}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(call, inputs))

    def sample_and_evaluate(self, current_prompt, sample_pool:DynamicSamplePool):
        if sample_pool:
            samples = sample_pool.sample(type=SampleType.Negative, k=self.task.config.batch_size)
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

        if sample_pool:
            for s, out, gold in zip(samples, outputs, golds):
                reward = 1 if self.task.get_reward(out, gold) else 0
                sample_pool.add_or_update(s, reward)

        return samples, inputs, outputs, golds

    def format_wrong_examples(self, inputs, outputs, golds):
        return "\n\n".join([
            f"[Example {i+1}]\n"
            f"Model's Input: \n{inp}\n"
            f"Model's Output: \n{out}\n"
            f"Model's Final Anwser: {self.task._normalize_answer(out)}\n"
            f"Expected Gold Anwser: {self.task._normalize_answer(gold)}\n"
            for i, (inp, out, gold) in enumerate(zip(inputs, outputs, golds))
            if self.task.get_reward(out, gold) == 0
        ])

    def analyze_and_rewrite(self, current_prompt, wrong_examples, trajectory_prompts):
        evaluation = (
            "I'm writing prompts for a language model designed for a task.\n"

            "###  Information about the current prompt: ### \n"
            f"My current prompt:\n {current_prompt}\n\n"
            f"This prompt leads to incorrect responses for the following examples:\n {wrong_examples}\n\n"

            "###  Analysis of Failure Cases: ### \n"
            "For each wrong example, you should carefully analyze why model response wrong answer, why my prompt leads to wrong answer.\n"
            "Provide comprehensive analysis of the common failure modes, pitfalls, or ambiguities in the prompt that may have contributed to these errors.\n"
            "List me all suggest improvements to the prompt to ensure better generalization.\n"
        )
        analysis = self.analyzer_model.api_call(evaluation)

        trajectory_prompts_str = "\n".join(
            [f"[Prompt {i+1}]: {p}" for i, p in enumerate(trajectory_prompts)]
        )

        rewriting_input = (
            "I’m optimizing a prompt for a language model on a specific task.\n"
            
            "###  Information about the current prompt: ### \n"
            f"My current prompt:\n {current_prompt}\n\n"
            f"This prompt leads to incorrect responses for the following examples:\n {wrong_examples}\n\n"
            f"Some analysis and suggestions for avoid wrong answers:\n {analysis}\n\n"
            f"There are a list of former prompts evolve to the current prompt, and each prompt is modified from its former prompts:\n{trajectory_prompts_str}\n\n"
            
            "### Requirements for the new prompt: ### \n"
            "The new prompt should solve the current prompt's problems.\n"
            "The new prompt should consider the list of prompts and evolve based on the current prompt.\n"
            "Please rewrite the prompt accordingly. Only output the new prompt.\n"

            "### Suggestions: ###\n"
            "1. You may consider adding similar failure examples as 'few-shot' references to help the model avoid repeating similar mistakes.\n"
            "2. You could reinforce reasoning steps that were missed in previous iterations, such as [specific reasoning step].\n"
            "3. There might be value in avoiding common mistakes such as [specific mistake, e.g., ambiguity in instructions].\n"
            "4. Providing additional context or clarifications may help the model better understand edge cases and task-specific details.\n"
            "5. You could focus on guiding the model step by step to improve its generalization ability in future tasks."
        )
        return self.rewriter_model.api_call(rewriting_input)

    def do(self, current_prompt, trajectory_prompts, sample_pool=None):
        attempts = 0
        wrong_examples = None
        while not wrong_examples and attempts < self.max_resample_attempts:
            _, inputs, outputs, golds = self.sample_and_evaluate(current_prompt, sample_pool)
            wrong_examples = self.format_wrong_examples(inputs, outputs, golds)
            attempts += 1

        if not wrong_examples:
            self.sample_failure_counter += 1
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
        self.sample_failure_counter = 0  # 连续失败次数

    def _batch_api_call(self, inputs: list):
        def call(x):
            try:
                return self.tester_model.api_call(x)
            except Exception as e:
                return f"[ERROR] {str(e)}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            return list(executor.map(call, inputs))

    def sample_and_evaluate(self, current_prompt, sample_pool:DynamicSamplePool):

        if sample_pool:
            samples = sample_pool.sample(type=SampleType.Positive, k=self.task.config.batch_size)
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

        if sample_pool:
            for s, out, gold in zip(samples, outputs, golds):
                reward = 1 if self.task.get_reward(out, gold) else 0
                sample_pool.add_or_update(s, reward)

        return samples, inputs, outputs, golds

    def format_success_examples(self, inputs, outputs, golds):
        correct_blocks = []
        for _, (inp, out, gold) in enumerate(zip(inputs, outputs, golds)):
            if self.task.get_reward(out, gold) == 1:
                correct_blocks.append(
                    f"[Example {len(correct_blocks)+1}]\n"
                    f"Model's Input: \n{inp}\n"
                    f"Model's Output: \n{out}\n"
                    f"Model's Final Answer: {self.task._normalize_answer(out)}\n"
                    f"Expected Gold Answer: {self.task._normalize_answer(gold)}"
                )
        return "\n\n".join(correct_blocks)

    def analyze_and_rewrite(self, current_prompt, success_examples, trajectory_prompts):
        evaluation = (
            "I'm optimizing a prompt for a language model on a specific task.\n"

            "###  Information about the current prompt: ### \n"
            f"My current prompt:\n{current_prompt}\n\n"
            f"Here are some successful examples where the model's prediction matches the correct answer:\n{success_examples}\n\n"
            
            "###  Analysis of Successful Cases: ### \n"
            "For each example, analyze why the model succeeded. \n"
            "Summarize the key reasoning strategies, invariants, decision rules, or intermediate steps "
            "that should be reinforced in the prompt to generalize better."
        )
        analysis = self.reasoning_model.api_call(evaluation)

        trajectory_prompts_str = "\n".join(
            [f"[Prompt {i+1}]: {p}" for i, p in enumerate(trajectory_prompts)]
        )

        rewriting_input = (
            "I'm optimizing a prompt for a language model.\n"

            "### Information about the current prompt: ### \n"
            f"My current prompt:\n{current_prompt}\n\n"
            f"Here are some successful examples where the model's prediction matches the correct answer:\n{success_examples}\n\n"
            f"The following strengths and reasoning strategies were identified from successful examples:\n{analysis}\n\n"
            f"There are a list of former prompts evolve to the current prompt, and each prompt is modified from its former prompts:\n{trajectory_prompts_str}\n\n"
            
            "###  Requirements for the new prompt: ### \n"
            "The new prompt should consider the list of prompts and evolve based on the current prompt.\n"
            "Please rewrite the prompt to incorporate and emphasize these strengths while keeping it concise and clear.\n"
            "Only output the new prompt.\n\n"

            "### Suggestions: ###\n"
            "1. You might want to reinforce successful reasoning steps that contributed to the correct answers (e.g., [specific reasoning step]).\n"
            "2. You could add 'few-shot' examples based on successful cases to guide the model in future tasks, especially for [specific type of task].\n"
            "3. It might be helpful to keep the prompt concise while emphasizing the key reasoning strategies that worked in these examples.\n"
            "4. You may want to ensure that the model can generalize these strengths to new tasks or edge cases.\n"
            "5. You could highlight decision rules or key steps that contributed to success, and ensure they are emphasized in the new prompt."
        )
        return self.rewriter_model.api_call(rewriting_input)

    def do(self, current_prompt, trajectory_prompts, sample_pool=None):
        attempts = 0
        success_examples_text = None
        while not success_examples_text and attempts < self.max_resample:
            _, inputs, outputs, golds = self.sample_and_evaluate(current_prompt, sample_pool)
            success_examples_text = self.format_success_examples(inputs, outputs, golds)
            attempts += 1

        if not success_examples_text:
            self.sample_failure_counter += 1
            return current_prompt

        rewritten_prompt = self.analyze_and_rewrite(current_prompt, success_examples_text, trajectory_prompts)
        super().do(rewritten_prompt, trajectory_prompts, sample_pool=sample_pool)
        return rewritten_prompt

    
def define_full_actions(task: TaskBase):
    return [
        FailureDrivenAction(task),
        SuccessDrivenAction(task),
    ]

def define_failure_actions(task: TaskBase):
    return [
        FailureDrivenAction(task),
    ]