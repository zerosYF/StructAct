import os
from search.evaluator import PromptEvaluator
from search.config import SearchConfig
from task.bbh.geometric_shapes import GeometricShapesTask
pag_prompt = """
"""
with open(os.path.join("promptagent_test/promptagent_geo.txt"), "r", encoding="utf-8") as f:
    pag_prompt = f.read()

config = SearchConfig()
task = GeometricShapesTask(config)
evaluator = PromptEvaluator(task, config.reward_thread_num)
acc_pag = evaluator.evaluate(task.get_test(), pag_prompt)
print(acc_pag.get("accuracy"))