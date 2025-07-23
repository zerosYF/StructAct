from task.epistemic import EpistemicTask
from search.controller import SearchController
from search.config import SearchConfig
from search.evaluator import PromptEvaluator
from logger import logger
import time

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run():
    config = SearchConfig()
    bbh_task = EpistemicTask(config=config)
    evaluator = PromptEvaluator(bbh_task, config.reward_thread_num)
    controller = SearchController(evaluator, config, bbh_task)
    logger.info("🔍 Training to get best prompt...")

    start_time = time.time()
    best_template, best_prompt = controller.search()
    end_time = time.time()
    duration = end_time - start_time  # 单位：秒
    minutes, seconds = divmod(duration, 60)
    logger.info(f"✅ Finished search in {int(minutes)} min {int(seconds)} sec")

    logger.info(f"✅ Best PromptTemplate:\n{best_template}")
    acc_mcts = evaluator.evaluate(bbh_task.get_test(), best_prompt)
    logger.info(f"📊 MCTS Test Accuracy:{acc_mcts.get('accuracy')}")
    acc_origin = evaluator.evaluate(bbh_task.get_test(), bbh_task.extract_origin_prompt())
    logger.info(f"📊 Original Test Accuracy:{acc_origin.get('accuracy')}")

 
if __name__ == "__main__":
    run()

