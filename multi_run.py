import os
import time
import argparse
import traceback
import multiprocessing
from logger import logger

from task.base_task import TaskBase
from task.causal_judgement import CausalJudgementTask
from task.epistemic import EpistemicTask
from task.geometric_shapes import GeometricShapesTask
from task.object_counting import ObjectCountingTask
from task.penguins_table import PenguinsTableTask
from task.temporal_sequences import TemporalSequencesTask
from task.gsm8k import GSM8KTask
from task.multi_arith import SimpleMathReasoningTask

from search.controller import SearchController
from search.config import SearchConfig
from search.evaluator import PromptEvaluator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ✅ 显式注册所有可用任务
TASK_REGISTRY = {
    "CausalJudgementTask": CausalJudgementTask,
    "EpistemicTask": EpistemicTask,
    "GeometricShapesTask": GeometricShapesTask,
    "ObjectCountingTask": ObjectCountingTask,
    "PenguinsTableTask": PenguinsTableTask,
    "TemporalSequencesTask": TemporalSequencesTask,
    "GSM8KTask": GSM8KTask,
    "SimpleMathReasoningTask": SimpleMathReasoningTask,
}


def run_task(task_name: str):
    config = SearchConfig()
    try:
        task_cls = TASK_REGISTRY[task_name]
        task: TaskBase = task_cls(config)
        evaluator = PromptEvaluator(task, config.reward_thread_num)
        controller = SearchController(evaluator, config, task)

        logger.info(f"🚀 Running task: {task.name}")
        start_time = time.time()

        best_template, best_prompt = controller.search()

        acc_mcts = evaluator.evaluate(task.get_test(), best_prompt)
        acc_origin = evaluator.evaluate(task.get_test(), task.extract_origin_prompt())

        end_time = time.time()
        duration = end_time - start_time
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
        logger.error(f"❌ Error in task {task_name}: {str(e)}")
        traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser(description="Run prompt search on selected tasks.")
    parser.add_argument(
        "--task", nargs="+", required=True, choices=TASK_REGISTRY.keys(),
        help="Tasks to run. Choose one or more from: " + ", ".join(TASK_REGISTRY.keys())
    )
    return parser.parse_args()


def run_all(task_names: list[str]):
    num_workers = min(len(task_names), multiprocessing.cpu_count())
    print(f"🚦 Running {len(task_names)} tasks using {num_workers} workers...\n")

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_task, task_names)

    print("🎉 All tasks completed.")


if __name__ == "__main__":
    args = parse_args()
    run_all(args.task)