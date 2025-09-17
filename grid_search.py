import json
import itertools
import logging
from search.dual_search import DualSearchController
from search.config import SearchConfig
from search.evaluator import PromptEvaluator
from task.bbeh.bool_expressions import BooleanExpressionsTask
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import copy

logger = logging.getLogger(__name__)

def worker(lam, theta, controller_class, evaluator, config, task, output_dir):
    # 每个线程都用自己的 config 副本，避免冲突
    from copy import deepcopy
    local_config = deepcopy(config)
    local_config.negative_var_mag = lam
    local_config.negative_informative_mag = theta

    controller = controller_class(evaluator, local_config, task)
    best_prompt = controller.search()[1]
    acc_sa = evaluator.evaluate(task.get_test(), best_prompt)

    result = {
        "lam": lam,
        "theta": theta,
        "avg_reward": acc_sa,
        "best_prompt": best_prompt
    }

    # 保存单个结果
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"lam{lam}_theta{theta}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result  # 如果你不想汇总，可以删掉这一行


def run_grid_search(controller_class, evaluator, config, task, 
                    lam_candidates, theta_candidates, output_dir="grid_search_results",
                    max_workers=4):
    """
    网格搜索负反馈参数 lam / theta (多线程版本)
    """
    tasks = list(itertools.product(lam_candidates, theta_candidates))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker, lam, theta, controller_class, evaluator, config, task, output_dir): (lam, theta)
            for lam, theta in tasks
        }

        for future in as_completed(futures):
            lam, theta = futures[future]
            try:
                result = future.result()
                print(f"✅ Finished lam={lam}, theta={theta}, avg_reward={result['avg_reward']:.4f}")
            except Exception as e:
                print(f"❌ Error in lam={lam}, theta={theta}: {e}")

    print(f"Grid search completed. Results saved to {output_dir}")


# --------------------------
# 使用示例
# --------------------------
# lam_candidates = [0.4]
# theta_candidates = [0.16]

lam_candidates = [0.4, 0.5, 0.6]
theta_candidates = [0.16, 0.2, 0.24]
my_config = SearchConfig()
my_task = BooleanExpressionsTask(my_config)
my_evaluator = PromptEvaluator(my_task, my_config.reward_thread_num)

results = run_grid_search(
    controller_class=DualSearchController,
    evaluator=my_evaluator,   # 你的 PromptEvaluator 实例
    config=my_config,         # 你的 SearchConfig 实例
    task=my_task,             # 你的 TaskBase 实例
    lam_candidates=lam_candidates,
    theta_candidates=theta_candidates,
)