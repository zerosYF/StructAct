from src.action.base_action import OptimizeAction
from src.evaluator import PromptEvaluator
from src.mcts.node import Node, Step
from src.pool.sample_pools import DynamicSamplePool
from src.logger import logger
from typing import List, Set
import numpy as np
import math

class PromptNode(Node):
    def __init__(self, 
                 action_set: Set[OptimizeAction],
                 action_seq: List[OptimizeAction], 
                 trajectory_prompts: List[str],
                 prompt: str, 
                 evaluator: PromptEvaluator, 
                 depth: int, 

                 sample_pool: DynamicSamplePool = None,

                 Q:float=0.0, N:int=0, 
                 uct_value:float=0.0, 
                 parent=None
                 ):
        super().__init__(depth=depth, Q=Q, N=N, uct_value=uct_value, parent=parent)
        self.type = prompt
        self.action_set: Set[OptimizeAction] = action_set
        self.evaluator: PromptEvaluator = evaluator
        self.trajectory_prompts: List[str] = trajectory_prompts
        self.current_prompt: str = prompt
        logger.info(f"📜 Get new node at depth {depth} using action {action_seq[-1].name if len(action_seq) > 0 else None}")
        
        self.action_seq: List[OptimizeAction] = action_seq
        self.pool = sample_pool
        self.reward_value: float = self.reward()
        self.Q = self.reward_value

    
    def _weighted_random_choice(self, alpha=1.0, temperature=1.0) -> OptimizeAction:
        def _softmax(vals: np.ndarray, temperature: float):
            vals = vals - np.max(vals)  # 防止 exp 溢出
            exp_vals = np.exp(vals / max(temperature, 1e-6))
            probs = exp_vals / np.sum(exp_vals)
            return probs

        """
        动作选择：融合失败/使用次数和样本池状态（cpool diagnostics），带平滑。
        逻辑：
        - 基础项：历史采样失败次数和使用次数降低概率（探索/避免重复）
        - 样本池 bias：根据池子中成功/失败样本比例调节 Positive / Negative 动作偏好
        - 极端情况抑制：当池子中成功/失败样本极少时，抑制对应动作
        """
        actions: list[OptimizeAction] = list(self.action_set)

        # 基础项: failure + usage 抑制
        failure_counts = np.array([a.sample_failure_counter for a in actions])
        usage_counts = np.array([a.usage_count for a in actions])
        base_logits = - (failure_counts + usage_counts)

        # 样本池状态，可选
        action_bias = np.zeros(len(actions))
        if self.pool is not None:
            cpool_diag = self.pool.compute_cpool()
            easy_ratio = cpool_diag.get("easy_ratio", 0.0)
            informative_ratio = cpool_diag.get("informative_ratio", 0.0)
            hard_ratio = cpool_diag.get("hard_ratio", 0.0)

            success_ratio = easy_ratio + informative_ratio
            failure_ratio = hard_ratio

            # 用平滑函数 (tanh) 计算差异偏置
            diff = success_ratio - failure_ratio
            # 曲线陡峭度随样本池规模调整
            scale = math.log1p(cpool_diag.get("total_samples", 0.0))
            smooth_diff = np.tanh(diff * scale)
            # 样本池越大，说明统计结果越可靠，就可以更“自信”地放大 diff 值

            entropy = - (
                success_ratio * math.log(success_ratio + 1e-6) +
                failure_ratio * math.log(failure_ratio + 1e-6)
            )

            for i, a in enumerate(actions):
                if a.name.lower().startswith("SuccessDrivenAction"):
                    action_bias[i] = entropy * smooth_diff
                elif a.name.lower().startswith("FailureDrivenAction"):
                    action_bias[i] = - entropy * smooth_diff

        # 融合
        logits = base_logits + alpha * action_bias
        probs = _softmax(logits, temperature)
        selected_index = np.random.choice(len(actions), p=probs)
        return actions[selected_index]
    
    def get_exploration_weight(self, exploration_weight=1.41):
        def cpool_mapping(cpool: float, k: float = 3.0) -> float:
            """
            将 cpool 值 (0~1) 平滑映射到 [0.5, 2] 范围
            - cpool 越大，探索系数越大
            - k 控制增长的快慢
            """
            return 0.5 + 1.5 * (1 - np.exp(-k * cpool))
        if self.pool:
            return cpool_mapping(self.pool.compute_cpool()['cpool'])
        return super().get_exploration_weight(exploration_weight)

    def take_action(self, step_type:Step):
        # Then apply the strategy-level semantic transformation.
        params_bundle = self.pool.get_net_controller().predict_and_apply()
        action:OptimizeAction = self._weighted_random_choice(params_bundle.get_mcts_alpha().item())
        new_prompt = action.do(
            current_prompt=self.current_prompt, 
            trajectory_prompts=self.trajectory_prompts, 
            sample_pool=self.pool
            )
        logger.info(f"📊 Current Step Type:{step_type}")
        logger.info(f"📊 Current Prompt:\n{new_prompt}")
        new_child = PromptNode(
            action_set=self.action_set,
            action_seq=self.action_seq + [action],
            trajectory_prompts=self.trajectory_prompts + [self.current_prompt],
            prompt=new_prompt,
            evaluator=self.evaluator,
            sample_pool=self.pool,

            depth=self.depth + 1,
            parent=self,
        )
        return new_child

    def reward(self):
        val_samples = self.evaluator.task.get_eval()
        score = self.evaluator.batch_reward(self.current_prompt, val_samples)
        logger.info(f"🎯 [Reward] Prompt evaluation score = {score:.4f}, Action sequence = {[a.name for a in self.action_seq]}")
        self.pool.get_net_controller().reforce(score)
        return score
    
    def q_value(self, last_q, rollout_reward):
        return last_q + rollout_reward
    
    def serialize(self, node_id:int):
        node_dict = {
            "id": node_id,
            "depth": self.depth,
            "action_sequence": [a.name for a in self.action_seq],
            "prompt": self.current_prompt,
            "Q": self.Q,
            "N": self.N, 
            "uct_value": self.uct_value,
            "reward": self.reward_value,
            "children": []
        }

        for child in self.children:
            node_id += 1
            child_serialized = child.serialize(node_id)
            if child_serialized:
                node_dict["children"].append(child_serialized)

        return node_dict