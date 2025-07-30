import os
from search.evaluator import PromptEvaluator
from search.config import SearchConfig
from task.bbh.epistemic import EpistemicTask
pag_prompt = """
Determine if a hypothesis logically follows from a given premise.

Act as a Logical Analyst, focusing on the context and details provided in the premise and hypothesis.

**Examples:**

1. **Input:**
   - Premise: Charles believes that two teams of men are playing basketball on a court in an empty stadium.
   - Hypothesis: Ava believes that the men are playing basketball with no audience.
   - **Analysis:** The premise states Charles's belief, while the hypothesis refers to Ava's belief. There is no information about Ava's beliefs in the premise.
   - **Output:** non-entailment

2. **Input:**
   - Premise: Richard learns that Isabella thinks that a man looks mysterious in a blue shirt and a red truck for a company called Wilbert.
   - Hypothesis: Isabella thinks that a man looks mysterious in a blue shirt and a red truck for a company called Wilbert.
   - **Analysis:** The premise directly states Isabella's thoughts, which are identical to the hypothesis.
   - **Output:** entailment

3. **Input:**
   - Premise: Olivia knows that people are checking out a car with all its doors open in a parking lot in the city.
   - Hypothesis: Olivia knows that a car door has not been shut.
   - **Analysis:** If all the doors are open, it logically follows that at least one door has not been shut.
   - **Output:** entailment

No additional constraints are applied.

**Cautions:**
1. Keep your reasoning clear and consistent.
2. Do not introduce new information in the hypothesis that isn’t in the premise.
3. Analyze step by step to avoid logical errors.
4. Ensure the hypothesis is a direct and necessary result of the premise.
5. Pay close attention to the logical implications of nested beliefs and assumptions.
"""
# with open(os.path.join("promptagent_eps.txt"), "r", encoding="utf-8") as f:
#     pag_prompt = f.read()

config = SearchConfig()
task = EpistemicTask(config)
evaluator = PromptEvaluator(task, config.reward_thread_num)
acc_pag = evaluator.evaluate(task.get_test(), pag_prompt)
print(acc_pag.get("accuracy"))