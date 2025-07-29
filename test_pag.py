import os
from search.evaluator import PromptEvaluator
from search.config import SearchConfig
from task.bbh.epistemic import EpistemicTask
pag_prompt = ""
with open(os.path.join("promptagent_eps.txt"), "r", encoding="utf-8") as f:
    pag_prompt = f.read()

config = SearchConfig()
task = EpistemicTask(config)
evaluator = PromptEvaluator(task, config.reward_thread_num)
acc_pag = evaluator.evaluate(task.get_test(), pag_prompt)
print(acc_pag.get("accuracy"))