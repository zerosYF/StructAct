from task.big_bench_hard import BBHTask
from mcts.mcts_controller import MCTSController
from search.config import SearchConfig
from search.evaluator import PromptEvaluator
from model.model import Model, model
from search.action import define_full_actions
from logger import logger

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run():
    config = SearchConfig()
    bbh_task = BBHTask(split_ratio=config.split_ratio, seed=config.shuffle_seed)
    evaluator = PromptEvaluator(bbh_task, config.few_shot_num, config.reward_thread_num)
    action_set = define_full_actions(bbh_task)
    controller = MCTSController(evaluator, config, bbh_task, action_set)
    logger.info("🔍 Training best action sequence...")
    best_template, best_sequence, best_prompt = controller.search()
    logger.info(f"✅ Best PromptTemplate:\n{best_template}")
    logger.info(f"✅ Best ActionSequence:\n{chr(10).join([action.name for action in best_sequence])}")
    acc_mcts = evaluator.evaluate(bbh_task.get_test(), best_prompt)
    logger.info(f"📊 MCTS Test Accuracy:{acc_mcts.get("accuracy")}")
    acc_origin = evaluator.evaluate(bbh_task.get_test(), bbh_task.extract_origin_prompt())
    logger.info(f"📊 Original Test Accuracy:{acc_origin.get("accuracy")}")

 
if __name__ == "__main__":
    run()

