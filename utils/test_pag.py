import os
from src.evaluator import PromptEvaluator
from src.config import SearchConfig
from task.bbeh.bool_expressions import BooleanExpressionsTask
pag_prompt = """
"""
with open(os.path.join("test_prompt.txt"), "r", encoding="utf-8") as f:
    pag_prompt = f.read()

config = SearchConfig()
task = BooleanExpressionsTask(config)
evaluator = PromptEvaluator(task, config.reward_thread_num)
acc_pag = evaluator.evaluate(task.get_test(), pag_prompt)
print(acc_pag.get("accuracy"))