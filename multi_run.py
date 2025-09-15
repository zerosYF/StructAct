import os
import time
import argparse
import traceback
import multiprocessing
from logger import logger

from task.base_task import TaskBase
from task.bbh.causal_judgement import CausalJudgementTask
from task.bbh.epistemic import EpistemicTask
from task.bbh.geometric_shapes import GeometricShapesTask
from task.bbh.object_counting import ObjectCountingTask
from task.bbh.penguins_table import PenguinsTableTask
from task.bbh.temporal_sequences import TemporalSequencesTask
from task.math.gsm8k import GSM8KTask
from task.math.multi_arith import SimpleMathReasoningTask
from task.mmlu.ethos import HateSpeechDetectionTask
from task.mmlu.mmlu_business import BusinessMCQTask
from task.mmlu.mmlu_engineering import EngineeringMCQTask
from task.mmlu.med_qa import MedicalExamTask
from task.mmlu.case_hold import LegalHoldingTask
from task.mmlu.arc import PhysicsMCQTask
from task.bbeh.causal_understanding import CausalUnderstandingTask 
from task.bbeh.geometric_shapes import GeometricShapesTask as BGeometricShapesTask
from task.bbeh.temporal_sequence import TemporalSequenceTask as BTemporalSequenceTask
from task.bbeh.object_counting import ObjectCountingTask as BObjectCountingTask
from task.bbeh.bool_expressions import BooleanExpressionsTask

from search.dual_search import DualSearchController
from search.beam_search import BeamSearchController
from search.bline_search import PromptAgentController
from search.config import SearchConfig
from search.evaluator import PromptEvaluator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TASK_REGISTRY = {
    "CausalJudgementTask": CausalJudgementTask,
    "EpistemicTask": EpistemicTask,
    "GeometricShapesTask": GeometricShapesTask,
    "ObjectCountingTask": ObjectCountingTask,
    "PenguinsTableTask": PenguinsTableTask,
    "TemporalSequencesTask": TemporalSequencesTask,
    "GSM8KTask": GSM8KTask,
    "SimpleMathReasoningTask": SimpleMathReasoningTask,
    "HateSpeechDetectionTask": HateSpeechDetectionTask,
    "BusinessMCQTask": BusinessMCQTask,
    "EngineeringMCQTask":EngineeringMCQTask,
    "MedicalExamTask": MedicalExamTask,
    "LegalHoldingTask": LegalHoldingTask,
    "PhysicsMCQTask": PhysicsMCQTask,
    "CausalUnderstandingTask": CausalUnderstandingTask,
    "BGeometricShapesTask": BGeometricShapesTask,
    "BTemporalSequenceTask": BTemporalSequenceTask,
    "BObjectCountingTask":BObjectCountingTask,
    "BooleanExpressionsTask":BooleanExpressionsTask,
}


CONTROLLER_REGISTRY = {
    "PromptAgentController": PromptAgentController,
    "DualSearchController": DualSearchController,
    "BeamSearchController": BeamSearchController,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run prompt search on selected tasks.")
    parser.add_argument(
        "--task", nargs="+", required=True, choices=TASK_REGISTRY.keys(),
        help="Tasks to run. Choose one or more from: " + ", ".join(TASK_REGISTRY.keys())
    )
    parser.add_argument(
        "--controller", required=True, choices=CONTROLLER_REGISTRY.keys(),
        help="Controller to use for search."
    )
    return parser.parse_args()


def run_task(args):
    task_name, controller_name = args
    config = SearchConfig()
    try:
        task_cls = TASK_REGISTRY[task_name]
        task: TaskBase = task_cls(config)
        evaluator = PromptEvaluator(task, config.reward_thread_num)
        controller_cls = CONTROLLER_REGISTRY[controller_name]
        controller = controller_cls(evaluator, config, task)

        logger.info(f"🚀 Running task: {task.name} with {controller_name}")
        start_time = time.time()

        best_template, best_prompt = controller.search()

        acc_sa = evaluator.evaluate(task.get_test(), best_prompt)

        end_time = time.time()
        duration = end_time - start_time
        minutes, seconds = divmod(duration, 60)

        result_dir = os.path.join("results", f"{task_name}/{controller_name}")
        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(f"🔍 Task: {task.name}\n")
            f.write(f"✅ Best Prompt Template:\n{best_template}\n\n")
            f.write(f"✅ Best Prompt:\n{best_prompt}\n\n")
            f.write(f"📊 SA Test Accuracy: {acc_sa.get('accuracy')}\n")
            f.write(f"\n⏱️ Time Elapsed: {int(minutes)} min {int(seconds)} sec ({duration:.2f} seconds)\n")

        logger.info(f"✅ Finished task: {task.name} in {int(minutes)} min {int(seconds)} sec")

    except Exception as e:
        logger.error(f"❌ Error in task {task_name}: {str(e)}")
        traceback.print_exc()


def run_all(task_names: list[str], controller_name: str):
    num_workers = min(len(task_names), multiprocessing.cpu_count())
    print(f"🚦 Running {len(task_names)} tasks with {controller_name} using {num_workers} workers...\n")

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_task, [(task_name, controller_name) for task_name in task_names])

    print("🎉 All tasks completed.")


if __name__ == "__main__":
    args = parse_args()
    run_all(args.task, args.controller)