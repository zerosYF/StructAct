from task.base_task import TaskBase
from model.model import Model, getModel
from typing import List
from concurrent.futures import ThreadPoolExecutor
from logger import logger

class PromptEvaluator:
    def __init__(self, task:TaskBase, few_shot_num:int, thread_num:int):
        self.model:Model = getModel()
        self.task = task
        self.few_shot_num = few_shot_num
        self.thread_num = thread_num
    
    def reward(self, current_prompt:str, sample:dict) -> float:
        prompt = current_prompt
        #logger.info(f"reward prompt:{prompt}")
        output = self.model.api_call(self.task.system_prompt, prompt)
        logger.info(f"reward model answer:{output.strip().lower()}")
        logger.info(f"reward gold answer:{sample["answer"].strip().lower()}")
        return 1.0 if output.strip().lower() == sample["answer"].strip().lower() else 0.0
    
    def batch_reward(self, current_prompt: str, samples: List[dict]) -> List[float]:
        def _reward_one(s):
            output = self.model.api_call(self.task.system_prompt, current_prompt)
            logger.info(f"reward model answer:{output.strip().lower()}")
            logger.info(f"reward gold answer:{s['answer'].strip().lower()}")
            return 1.0 if output.strip().lower() == s["answer"].strip().lower() else 0.0

        with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            return list(executor.map(_reward_one, samples))

    def evaluate(self, test_data: list[dict], final_prompt: str) -> dict:
        total = len(test_data)

        def _evaluate_one(item):
            output = self.model.api_call(self.task.system_prompt, final_prompt)
            gold = item["answer"]
            correct = int(output.strip().lower() == gold.strip().lower())
            return {
                "prompt": final_prompt,
                "output": output,
                "answer": gold,
                "correct": correct,
            }

        # 并行处理所有样本
        with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            results = list(executor.map(_evaluate_one, test_data))

        correct = sum(r["correct"] for r in results)
        outputs = [{"prompt": r["prompt"], "output": r["output"], "answer": r["answer"]} for r in results]

        return {
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "outputs": outputs
        }