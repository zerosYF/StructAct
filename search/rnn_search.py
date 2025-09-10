from task.base_task import TaskBase
from search.evaluator import PromptEvaluator
from search.config import SearchConfig
from typing import List, Set
from visualizer import RNNVisualizer
from structs.prompt_template import PromptTemplate
from structs.mini_blocks import get_all_blocks
from search.search import SearchController
from logger import logger


class RNNSearchController(SearchController):
    def __init__(self,
                 evaluator: PromptEvaluator,
                 config: SearchConfig,
                 task: TaskBase):
        super().__init__(evaluator, config, task)
        self.controller = None

    def search(self):
        template = PromptTemplate(config=self.config, blocks=get_all_blocks(self.config), task=self.task)
        # Pass evaluator to template immediately so it's available for pre_sample
        template.evaluator = self.evaluator
        # Connect the SearchController to the TemplateController
        if self.controller is None:
            self.controller = template.controller
        logger.info(f"🔍 Initial template constraints:\n{template.describe()}")
        init_prompt = self.task.origin_prompt

        visualizer = RNNVisualizer()
        visualizer.start()

        # PPO 批量/小批参数（若 config 没有则走默认）
        ppo_batch_min  = getattr(self.config, "ppo_batch_min", 64)
        ppo_batch_max  = getattr(self.config, "ppo_batch_max", 512)
        ppo_minibatch  = getattr(self.config, "ppo_minibatch_size", 1) # 128
        ppo_target_kl  = getattr(self.config, "ppo_target_kl", 0.02)

        optimized_prompt = init_prompt

        for epoch in range(self.config.rnn_iter_num):
            logger.info(f"\n=== RNN Training Iteration {epoch+1}/{self.config.rnn_iter_num} ===")
            ppo_batch = 5
            # 本轮批量大小用你的非线性调度
            # ppo_batch = self.nonlinear_schedule(
            #     rnn_iter=epoch,
            #     total_iter=self.config.rnn_iter_num,
            #     min_val=ppo_batch_min,
            #     max_val=ppo_batch_max,
            #     steepness=8.0,
            #     pivot=0.85
            # )
            logger.info(f"🌀 PPO batch size for this epoch: {ppo_batch}")

            batch_samples = []
            prompt_pool: List[tuple] = []  # (prompt_text, reward, params)

            # === 多次 rollout：沿用你现有的 template.pre_sample(init_prompt) 接口（最小改动）===
            # 约定：pre_sample 内部会驱动 controller.sample，并返回 (prompt, reward)
            for _ in range(ppo_batch):
                sync_prompt, history_reward = template.pre_sample(init_prompt)
                # 采样完成后，controller 内部已缓存 last_params / last_log_prob_sum / last_value
                params = self.controller.last_params if self.controller is not None else None
                old_logp = self.controller.last_log_prob_sum if self.controller is not None else None
                old_value = self.controller.last_value if self.controller is not None else None

                if params is None or old_logp is None or old_value is None:
                    # 如果没有注入 controller，则退化为非 PPO（只更新模板与可视化）
                    prompt_pool.append((sync_prompt, float(history_reward), None))
                    continue

                batch_samples.append({
                    "params": params,
                    "old_logp": old_logp.detach(),
                    "old_value": old_value.detach().reshape(()),
                    "reward": float(history_reward),
                })
                prompt_pool.append((sync_prompt, float(history_reward), params))

            # === 批量 PPO 更新 ===
            if self.controller is not None and len(batch_samples) > 0:
                self.controller.reinforce_batch(
                    batch_samples,
                    minibatch_size=ppo_minibatch,
                    target_kl=ppo_target_kl
                )

            # === 选择本轮最优 prompt 并回写 ===
            prompt_pool.sort(key=lambda x: x[1], reverse=True)
            optimized_prompt, best_reward, best_params = prompt_pool[0]
            mean_reward = sum(r for _, r, _ in prompt_pool) / max(1, len(prompt_pool))

            logger.info(f"🏅 Epoch {epoch+1}: best_reward={best_reward:.4f} mean_reward={mean_reward:.4f}")
            # Use batch_entropy if available (from reinforce_batch), otherwise use last_entropy
            entropy_value = float(getattr(self.controller, "batch_entropy", getattr(self.controller, "last_entropy", 0.0)))
            visualizer.log_train(mean_reward, entropy_value)

            # Skip reinforce if batch update was already done
            skip_reinforce = self.controller is not None and len(batch_samples) > 0
            try:
                template.update(self.evaluator, optimized_prompt, history_reward=best_reward, skip_reinforce=skip_reinforce)
            except TypeError:
                # Fallback for older version without skip_reinforce parameter
                if not skip_reinforce:
                    template.update(self.evaluator, optimized_prompt)

        return template.describe(), optimized_prompt