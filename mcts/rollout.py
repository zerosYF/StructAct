from abc import ABC, abstractmethod
import random
from search.config import SearchConfig
from concurrent.futures import ThreadPoolExecutor
from search.evaluator import PromptEvaluator
from Experiment.mcts.prompt_node import PromptNode
from logger import logger

class RolloutStrategy(ABC):
    @abstractmethod
    def rollout(self, node) -> float:
        """从给定节点执行 rollout 并返回最终 reward"""
        pass

class MultiPathRollout(RolloutStrategy):
    def __init__(self, evaluator:PromptEvaluator, num_paths: int = 1, rollout_depth: int = 5):
        self.num_paths = num_paths
        self.rollout_depth = rollout_depth
        self.evaluator = evaluator

    def rollout(self, node:PromptNode) -> float:
        # 使用线程池并行执行 rollout
        with ThreadPoolExecutor(max_workers=self.num_paths) as executor:
            futures = [
                executor.submit(self._rollout_single_path, node.clone_node(), path_id)
                for path_id in range(self.num_paths)
            ]
            rewards = [f.result() for f in futures]

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        logger.info(f"🧠 多路径并行 rollout 平均 reward：{avg_reward:.4f}")
        return avg_reward
    
    def _rollout_single_path(self, node:PromptNode, path_id: int) -> float:
        current = node
        depth = 0
        rewards = []

        while depth < self.rollout_depth:
            actions = current.get_possible_actions()
            if not actions:
                break
            action = random.choice(actions)
            current = current.take_action(action)
            rewards.append(current.reward())

            avg_reward = sum(rewards) / len(rewards)
            logger.info(f"[路径 {path_id+1}] 第 {depth+1} 步动作: {getattr(action, 'name', '未知动作')}，Reward={avg_reward:.4f}")
            depth += 1

        return avg_reward if rewards else 0.0

def get_rollout_strategy(evaluator:PromptEvaluator, config:SearchConfig):
        if config.rollout_idx == 0:
            return MultiPathRollout(evaluator, config.rollout_path_num, config.rollout_length)
