import os
import traceback
from task.epistemic import EpistemicTask
from search.controller import SearchController
from search.config import SearchConfig
from search.evaluator import PromptEvaluator
from task.base_task import TaskBase
from logger import logger

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

config = SearchConfig()
def run_task(task:TaskBase):
    try:
        # 初始化任务、评估器和控制器
        bbh_task = task
        evaluator = PromptEvaluator(bbh_task, config.reward_thread_num)
        controller = SearchController(evaluator, config, bbh_task)

        logger.info(f"🚀 Running task: {task.name}")
        best_template, best_sequence, best_prompt = controller.search()

        acc_mcts = evaluator.evaluate(bbh_task.get_test(), best_prompt)
        acc_origin = evaluator.evaluate(bbh_task.get_test(), bbh_task.extract_origin_prompt())

        # 创建结果目录
        result_dir = os.path.join("results", task.name)
        os.makedirs(result_dir, exist_ok=True)

        # 写入结果
        with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(f"🔍 Task: {task.name}\n")
            f.write(f"✅ Best Prompt Template:\n{best_template}\n\n")
            f.write("✅ Best Action Sequence:\n")
            f.write("\n".join([action.name for action in best_sequence]) + "\n\n")
            f.write(f"📊 MCTS Test Accuracy: {acc_mcts.get('accuracy')}\n")
            f.write(f"📊 Original Test Accuracy: {acc_origin.get('accuracy')}\n")

        logger.info(f"✅ Finished task: {task.name}")

    except Exception as e:
        logger.error(f"❌ Error in task {task.name}: {str(e)}")
        traceback.print_exc()

import multiprocessing
from task.causal_judgement import CausalJudgementTask
from task.epistemic import EpistemicTask
from task.geometric_shapes import GeometricShapesTask
from task.object_counting import ObjectCountingTask
from task.penguins_table import PenguinsTableTask
from task.temporal_sequences import TemporalSequencesTask
# 替换为你要运行的子任务名称列表

TASK_LIST = [
    CausalJudgementTask(config=config),
    EpistemicTask(config=config),
    GeometricShapesTask(config=config),
    ObjectCountingTask(config=config),
    PenguinsTableTask(config=config), 
    TemporalSequencesTask(config=config),
]

def run_all():
    num_workers = min(len(TASK_LIST), multiprocessing.cpu_count())

    print(f"🚦 Starting {len(TASK_LIST)} tasks with {num_workers} workers...\n")
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_task, TASK_LIST)

    print("🎉 All tasks completed.")

if __name__ == "__main__":
    run_all()