from model.model import Model, getOptimModel
from task.base_task import TaskBase
from program.base_action import OptimizeAction
import concurrent.futures

# Preload models
tester_model = getOptimModel()
analyzer_model = getOptimModel()
rewriter_model = getOptimModel()
checker_model = getOptimModel()
builder_model = getOptimModel()
evaluator_model = getOptimModel()

class TestReflectRewriteAction(OptimizeAction):
    """Evaluate model output → reflect on failure → revise prompt"""
    def __init__(self, task, name="TestReflectRewriteAction"):
        super().__init__(task, name)
        self.tester_model = tester_model
        self.analyzer_model = analyzer_model
        self.rewriter_model = rewriter_model
        self.tester_system_prompt = (
            "You're a QA assistant. Given prompt and input, generate accurate answers."
        )
        self.analyzer_system_prompt = (
            "You are a reasoning evaluator. Given a list of input-output-gold triplets, "
            "analyze the model's correctness and reflect on reasons for success or failure.\n\n"
            "For **each example**, output the following format:\n"
            "1. [Example i]\n"
            "2. Input Summary: <Brief summary of the input>\n"
            "3. Output Evaluation: <Correct / Incorrect>\n"
            "4. Reasoning: <Why is the output correct or not>\n"
            "5. Suggestion: <What could be changed in the prompt to help>\n\n"
            "After going through all examples, output a final section:\n"
            "**Overall Reflection:**\n"
            "- Common errors:\n"
            "- Strengths:\n"
            "- Prompt improvement suggestions (be specific):\n"
        )
        self.rewriter_system_prompt = (
            "You are a prompt editor. "
            "You must strictly follow the given prompt structure. "
        )
    
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
        samples = self.task.sample_train_mcts(self.task.config.batch_size)
        inputs = [self.task.extract_tuple(s)[0] for s in samples]
        golds = [self.task.extract_tuple(s)[1] for s in samples]
        final_inputs = [self.task.inject_final_input(current_prompt, inp) for inp in inputs]
        outputs = self._batch_api_call(final_inputs)

        evaluation = "\n\n".join([
            f"[Example {i+1}]\n"
            f"Input: {inp}\n"
            f"Model Output: {out}\n"
            f"Expected Output: {gold}"
            for i, (inp, out, gold) in enumerate(zip(inputs, outputs, golds))
        ])
        evaluation = (
            "You should analyze why the model outputs the right or wrong answer, "
            "and suggest improvements to the prompt to ensure better generalization.\n"
            + evaluation
        )
        analysis = self.analyzer_model.api_call(self.analyzer_system_prompt, evaluation)

        prompt = (
            "<CurrentPrompt>\n"
            f"{current_prompt}\n"
            "</CurrentPrompt>\n\n"
            
            "<ExampleAnalysis>\n"
            f"{analysis}\n"
            "</ExampleAnalysis>\n\n"
            
            "<PromptStructure>\n"
            f"{template_description}\n"
            "</PromptStructure>\n\n"
            
            "<Instruction>\n"
            """
            - You must revise the <CurrentPrompt> to improve task performance.
            - Your revision must strictly follow the <PromptStructure>.
            - Use the findings in <ExampleAnalysis> to guide your revision.
            - Only output the revised prompt **without any explanation**.
            - Do not change the structure layout or add new sections."""
            "</Instruction>\n\n"
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, prompt)

class FewShotExampleBuilder(OptimizeAction):
    """Add new few-shot examples and integrate into prompt"""
    def __init__(self, task, name="FewShotExampleBuilder"):
        super().__init__(task, name)
        self.builder_model = builder_model
        self.rewriter_model = rewriter_model
        self.builder_system_prompt = "Generate 1–2 new few-shot QA pairs based on current prompt."
        self.rewriter_system_prompt = (
            "You are a prompt editor. "
            "You must strictly follow the given prompt structure. "
            "Replace some examples in the prompt, keeping structure consistent."
        )

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)
        original_examples = self.task.sample_train_mcts(self.task.config.batch_size)
        original_text = self.task.samples2text(original_examples)

        builder_input = (
            "<Prompt>\n"
            f"{current_prompt}\n"
            "</Prompt>\n\n"
            
            "<ExistingExamples>\n"
            f"{original_text}\n"
            "</ExistingExamples>\n\n"
            
            "<Instruction>\n"
            """
            Generate 1–2 **new high-quality few-shot QA pairs** that align with the current prompt style.
            Examples should be helpful for the model to perform better on this task.
            Do not repeat existing examples."""
            "</Instruction>\n"
        )
        new_examples = self.builder_model.api_call(self.builder_system_prompt, builder_input)
        new_examples += "\n" + original_text 

        rewriting_input = (
            "<CurrentPrompt>\n"
            f"{current_prompt}\n"
            "</CurrentPrompt>\n\n"

            "<CandidateExamples>\n"
            f"{new_examples}\n"
            "</CandidateExamples>\n\n"

            "<PromptStructure>\n"
            f"{template_description}\n"
            "</PromptStructure>\n\n"

            "<Instruction>\n"
            """
            - Revise the `<CurrentPrompt>` by updating its few-shot examples.
            - From `<CandidateExamples>`, select the **best k examples** (where k is determined by the structure).
            - You must output exactly K examples in the final prompt.
            - K is determined by the PromptStructure block.
            - Do not output fewer or more examples than expected.
            - You **can replace** some or all original examples.
            - The revised prompt **must strictly follow** the structure in `<PromptStructure>`.
            - Keep all other prompt components (e.g., role, instructions) unchanged.
            - Do **not** add explanations, analysis, or any other text.
            - Only output the **final revised prompt**."""
            "</Instruction>\n"
        )
        return self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_input)

class InstructionSimplifierByAbstraction(OptimizeAction):
    """Abstract task intent from examples and inject into prompt"""
    def __init__(self, task, name="InstructionSimplifierByAbstraction"):
        super().__init__(task, name)
        self.evaluator_model = evaluator_model
        self.rewriter_model = rewriter_model

        self.evaluator_system_prompt = (
            "You are an instruction summarizer.\n"
            "Given a few-shot QA dataset, abstract the overall task goal or instruction in 1–2 concise sentences.\n"
            "Focus on generalizing the underlying reasoning task, not surface content.\n"
            "Avoid referencing specific examples.\n"
        )

        self.rewriter_system_prompt = (
            "You are a prompt editor.\n"
            "Your task is to inject a more abstract, generalized task instruction into the prompt.\n"
            "You must **strictly follow the provided PromptStructure** — do not change its layout, section order, or any other content except the instruction block.\n"
            "Replace or refine the task instruction block with the new abstracted goal.\n"
            "Only output the **final revised prompt**, and nothing else.\n"
        )

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)

        samples = self.task.sample_train_mcts(self.task.config.batch_size)
        qa_text = self.task.samples2text(samples)
        summary_input = (
            "<FewShotExamples>\n"
            f"{qa_text}\n"
            "</FewShotExamples>\n\n"
            "<Instruction>\n"
            "Summarize the abstract task goal underlying these QA examples.\n"
            "Your output must be a short, generalized instruction.\n"
            "</Instruction>"
        )
        abstract_goal = self.evaluator_model.api_call(self.evaluator_system_prompt, summary_input)

        rewriting_input = (
            "<CurrentPrompt>\n"
            f"{current_prompt}\n"
            "</CurrentPrompt>\n\n"

            "<AbstractTaskGoal>\n"
            f"{abstract_goal}\n"
            "</AbstractTaskGoal>\n\n"

            "<PromptStructure>\n"
            f"{template_description}\n"
            "</PromptStructure>\n\n"

            "<Instruction>\n"
            """
            - Refine the current task instruction in <CurrentPrompt> with the abstract task goal from <AbstractTaskGoal>.\n
            - Follow the layout and section names defined in <PromptStructure> strictly.\n
            - Do not change the number of few-shot examples, roles, or any other part.\n
            - Output only the final revised prompt, without any extra explanation.\n"""
            "</Instruction>"
        )

        return self.rewriter_model.api_call(self.rewriter_system_prompt, rewriting_input)

class LexicalSimplifier(OptimizeAction):
    """Simplify wording to improve clarity and readability without changing structure."""
    def __init__(self, task, name="LexicalSimplifier"):
        super().__init__(task, name)
        self.rewriter_model = rewriter_model
        self.rewriter_system_prompt = (
            "You are a prompt editor.\n"
            "Your goal is to simplify wording for clarity and fluency.\n"
            "However, you must strictly preserve the original **structure**, section order, and semantics.\n"
            "Do not remove, reorder, or merge sections.\n"
            "Only revise the language for simplicity and readability."
        )

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)

        prompt = (
            "<CurrentPrompt>\n"
            f"{current_prompt}\n"
            "</CurrentPrompt>\n\n"

            "<PromptStructure>\n"
            f"{template_description}\n"
            "</PromptStructure>\n\n"

            "<Instruction>\n"
            """
            - Simplify complex wording in <CurrentPrompt>.\n
            - Avoid jargon, repetition, or overly long phrases.\n
            - You must not change the prompt's structure or its logical flow.\n
            - Use <PromptStructure> to ensure all sections are retained and respected.\n
            - Do not change the number or order of few-shot examples.\n
            - Do not alter formatting or section headers.\n
            - Output only the revised prompt — do not include explanation or metadata.\n"""
            "</Instruction>"
        )

        return self.rewriter_model.api_call(self.rewriter_system_prompt, prompt)

class CohesionImprover(OptimizeAction):
    """Improve the transitions and cohesion between different prompt blocks."""
    def __init__(self, task, name="CohesionImprover"):
        super().__init__(task, name)
        self.rewriter_model = rewriter_model
        self.rewriter_system_prompt = (
            "You are a prompt cohesion editor.\n"
            "Your goal is to improve the coherence and fluency between different prompt sections.\n"
            "Do NOT change the structure, content order, or formatting.\n"
            "Only adjust transitions, connective phrasing, or redundant gaps to make the prompt flow naturally.\n"
            "Preserve the identity and boundary of each section."
        )

    def do(self, current_prompt, template_description):
        super().do(current_prompt, template_description)

        prompt = (
            "<CurrentPrompt>\n"
            f"{current_prompt}\n"
            "</CurrentPrompt>\n\n"

            "<PromptStructure>\n"
            f"{template_description}\n"
            "</PromptStructure>\n\n"

            "<Instruction>\n"
            """
            - Improve only the linguistic cohesion and transitions between blocks in <CurrentPrompt>.\n
            - Do not change the section order, content meaning, or formatting.\n
            - Use <PromptStructure> to ensure the block layout is preserved.\n
            - Keep all blocks distinct and intact.\n
            - Add brief connective cues only where necessary (e.g., intro phrases, bridges).\n
            - Output ONLY the revised prompt — do not add explanations, comments, or any extra content.\n"""
            "</Instruction>"
        )

        return self.rewriter_model.api_call(self.rewriter_system_prompt, prompt)

    
def define_full_actions(task: TaskBase):
    return [
        TestReflectRewriteAction(task),
        FewShotExampleBuilder(task),
        InstructionSimplifierByAbstraction(task),
        LexicalSimplifier(task),
        CohesionImprover(task),
    ]