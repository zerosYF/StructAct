from abc import ABC, abstractmethod
import random
from search.config import SearchConfig
from concurrent.futures import ThreadPoolExecutor
from search.evaluator import PromptEvaluator
from mcts.node import Node
from logger import logger

class RolloutStrategy(ABC):
    @abstractmethod
    def rollout(self, node) -> float:
        """Perform a rollout starting from the given node and return the final reward"""
        pass

class MultiPathRollout(RolloutStrategy):
    def __init__(self, evaluator: PromptEvaluator, num_paths: int = 1, rollout_depth: int = 5):
        self.num_paths = num_paths
        self.rollout_depth = rollout_depth
        self.evaluator = evaluator

    def rollout(self, node: Node) -> float:
        # Use a thread pool to perform rollouts in parallel
        with ThreadPoolExecutor(max_workers=self.num_paths) as executor:
            futures = [
                executor.submit(self._rollout_single_path, node.clone_node(), path_id)
                for path_id in range(self.num_paths)
            ]
            rewards = [f.result() for f in futures]

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        logger.info(f"🧠 Multi-path parallel rollout average reward: {avg_reward:.4f}")
        return avg_reward

    def _rollout_single_path(self, node: Node, path_id: int) -> float:
        current: Node = node
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
            logger.info(f"[Path {path_id+1}] Step {depth+1} action: {getattr(action, 'name', 'Unknown action')}, Reward={avg_reward:.4f}")
            depth += 1

        return avg_reward if rewards else 0.0

def get_rollout_strategy(evaluator: PromptEvaluator, config: SearchConfig):
    if config.rollout_idx == 0:
        return MultiPathRollout(evaluator, config.rollout_path_num, config.rollout_length)
