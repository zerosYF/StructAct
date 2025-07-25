import os
import traceback
from task.epistemic import EpistemicTask
from search.controller import SearchController
from search.config import SearchConfig
from search.evaluator import PromptEvaluator
from task.base_task import TaskBase
from logger import logger
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

config = SearchConfig()
def run_task(task_cls):
    try:
        task:TaskBase = task_cls(config)
        evaluator = PromptEvaluator(task, config.reward_thread_num)
        controller = SearchController(evaluator, config, task)

        logger.info(f"🚀 Running task: {task.name}")
        start_time = time.time()
        best_template, best_prompt = controller.search()

        acc_mcts = evaluator.evaluate(task.get_test(), best_prompt)
        acc_origin = evaluator.evaluate(task.get_test(), task.extract_origin_prompt())

        end_time = time.time()
        duration = end_time - start_time  # 单位：秒
        minutes, seconds = divmod(duration, 60)

        result_dir = os.path.join("results", task.name)
        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(f"🔍 Task: {task.name}\n")
            f.write(f"✅ Best Prompt Template:\n{best_template}\n\n")
            f.write(f"📊 MCTS Test Accuracy: {acc_mcts.get('accuracy')}\n")
            f.write(f"📊 Original Test Accuracy: {acc_origin.get('accuracy')}\n")
            f.write(f"\n⏱️ Time Elapsed: {int(minutes)} min {int(seconds)} sec ({duration:.2f} seconds)\n")

        logger.info(f"✅ Finished task: {task.name} in {int(minutes)} min {int(seconds)} sec")

    except Exception as e:
        logger.error(f"❌ Error in task {task_cls.__name__}: {str(e)}")
        traceback.print_exc()

import multiprocessing
from task.causal_judgement import CausalJudgementTask
from task.epistemic import EpistemicTask
from task.geometric_shapes import GeometricShapesTask
from task.object_counting import ObjectCountingTask
from task.penguins_table import PenguinsTableTask
from task.temporal_sequences import TemporalSequencesTask
from task.gsm8k import GSM8KTask
from task.multi_arith import SimpleMathReasoningTask

TASK_LIST = [
    EpistemicTask,
    TemporalSequencesTask,
    ObjectCountingTask,
    CausalJudgementTask,
    GeometricShapesTask,
    PenguinsTableTask,
    GSM8KTask,
    SimpleMathReasoningTask,
]

def run_all():
    num_workers = min(len(TASK_LIST), multiprocessing.cpu_count())

    print(f"🚦 Starting {len(TASK_LIST)} tasks with {num_workers} workers...\n")
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_task, TASK_LIST)

    print("🎉 All tasks completed.")

if __name__ == "__main__":
    run_all()