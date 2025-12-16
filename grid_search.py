import json
import itertools
import logging
from search.dual_search import DualSearchController
from src.config import SearchConfig
from src.evaluator import PromptEvaluator
from task.bbeh.bool_expressions import BooleanExpressionsTask

logger = logging.getLogger(__name__)

def run_grid_search(controller_class, evaluator, config, task, 
                    lam_candidates, theta_candidates, output_dir="grid_search_results"):
    """
    网格搜索负反馈参数 lam / theta，每轮保存为单独文件
    controller_class: DualSearchController
    evaluator: PromptEvaluator 实例
    config: SearchConfig 实例
    task: TaskBase 实例
    lam_candidates, theta_candidates: 参数列表
    output_dir: 保存结果的文件夹
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for lam, theta in itertools.product(lam_candidates, theta_candidates):
        logger.info(f"Running grid search with lam={lam}, theta={theta}...")

        # 更新 config 中负反馈参数
        config.negative_var_mag = lam
        config.negative_informative_mag = theta

        # 初始化控制器
        controller = controller_class(evaluator, config, task)

        # 运行搜索
        best_prompt = controller.search()[1]  # search() 返回 ("", optimized_prompt)

        # 评估结果
        acc_sa = evaluator.evaluate(task.get_test(), best_prompt)

        # 保存结果
        result = {
            "lam": lam,
            "theta": theta,
            "avg_reward": acc_sa.get('accuracy'),
            "best_prompt": best_prompt
        }
        results.append(result)

        # 每轮写入单独文件
        file_name = f"grid_lam{lam}_theta{theta}.json"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Finished lam={lam}, theta={theta}, avg_reward={acc_sa.get('accuracy'):.4f}")
        logger.info(f"✅ Results saved to {file_path}")

    logger.info(f"✅ Grid search completed. All results saved in {output_dir}")
    return results


# --------------------------
# 使用示例
# --------------------------
# lam_candidates = [0.4]
# theta_candidates = [0.16]

# lam_candidates = [0.4]
# theta_candidates = [0.2]

# lam_candidates = [0.4]
# theta_candidates = [0.24]

# lam_candidates = [0.5]
# theta_candidates = [0.16]

# lam_candidates = [0.5]
# theta_candidates = [0.2]

# lam_candidates = [0.5]
# theta_candidates = [0.24]

# lam_candidates = [0.6]
# theta_candidates = [0.16]

# lam_candidates = [0.6]
# theta_candidates = [0.2]

lam_candidates = [0.6]
theta_candidates = [0.24]

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
    output_dir="negative_feedback_grid_search.json"
)