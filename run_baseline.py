# baseline_evaluator.py
import os
import time
import argparse
import traceback
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

from search.config import SearchConfig
from search.evaluator import PromptEvaluator

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
    "BTemporalSequenceTask":BTemporalSequenceTask,
    "BObjectCountingTask":BObjectCountingTask,
}


def evaluate_baselines(task: TaskBase, evaluator: PromptEvaluator):
    """计算任务的基础准确率（zero-shot / few-shot / cot 等）"""
    return {
        "acc_origin_zs": evaluator.evaluate(task.get_test(), task.generate_default_prompt()),
        "acc_origin_fs": evaluator.evaluate(task.get_test(), task.generate_fewshot_prompt()),
        "acc_origin_fs_": evaluator.evaluate(task.get_test(), task.generate_cotfewshot_prompt()),
        "acc_cot_zs": evaluator.evaluate(task.get_test(), task.generate_cot_format_prompt()),
        "acc_cot_fs": evaluator.evaluate(task.get_test(), task.generate_fewshot_and_cot_format_prompt()),
        "acc_cot_fs_": evaluator.evaluate(task.get_test(), task.generate_cotfewshot_and_cot_format_prompt()),
    }


def run_baseline(task_name: str):
    config = SearchConfig()
    try:
        task_cls = TASK_REGISTRY[task_name]
        task: TaskBase = task_cls(config)
        evaluator = PromptEvaluator(task, config.reward_thread_num)

        logger.info(f"🚀 Running baseline evaluation for task: {task.name}")
        start_time = time.time()

        baseline_results = evaluate_baselines(task, evaluator)

        end_time = time.time()
        duration = end_time - start_time
        minutes, seconds = divmod(duration, 60)

        result_dir = os.path.join("results", f"{task_name}/BaselineEvaluator")
        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(f"🔍 Task: {task.name}\n")
            f.write(f"📊 Original ZeroShot Test Accuracy: {baseline_results['acc_origin_zs'].get('accuracy')}\n")
            f.write(f"📊 Original FewShot Test Accuracy: {baseline_results['acc_origin_fs'].get('accuracy')}\n")
            f.write(f"📊 Original Cot_FewShot Test Accuracy: {baseline_results['acc_origin_fs_'].get('accuracy')}\n")
            f.write(f"📊 Cot ZeroShot Test Accuracy: {baseline_results['acc_cot_zs'].get('accuracy')}\n")
            f.write(f"📊 Cot FewShot Test Accuracy: {baseline_results['acc_cot_fs'].get('accuracy')}\n")
            f.write(f"📊 Cot Cot_FewShot Test Accuracy: {baseline_results['acc_cot_fs_'].get('accuracy')}\n")
            f.write(f"\n⏱️ Time Elapsed: {int(minutes)} min {int(seconds)} sec ({duration:.2f} seconds)\n")

        logger.info(f"✅ Finished baseline evaluation for task: {task.name} in {int(minutes)} min {int(seconds)} sec")

    except Exception as e:
        logger.error(f"❌ Error in baseline evaluation for task {task_name}: {str(e)}")
        traceback.print_exc()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline evaluation on selected tasks.")
    parser.add_argument(
        "--task", nargs="+", required=True, choices=TASK_REGISTRY.keys(),
        help="Tasks to run. Choose one or more from: " + ", ".join(TASK_REGISTRY.keys())
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for task_name in args.task:
        run_baseline(task_name)