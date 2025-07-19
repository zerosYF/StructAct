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

class ClassicPathRollout(RolloutStrategy):
    def __init__(self, evaluator: PromptEvaluator, rollout_depth: int = 5):
        self.rollout_depth = rollout_depth
        self.evaluator = evaluator

    def rollout(self, node: Node) -> float:
        current: Node = node.clone_node()
        depth = 0
        rewards = []

        while depth < self.rollout_depth:
            actions = current.get_possible_actions()
            if not actions:
                break
            action = random.choice(actions)
            current = current.take_action(action)

            reward = current.reward()
            rewards.append(reward)

            logger.info(f"[Rollout] Step {depth+1}, Action: {getattr(action, 'name', 'Unknown')}, Reward: {reward:.4f}")
            depth += 1

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        logger.info(f"🧠 Single-path rollout average reward: {avg_reward:.4f}")
        return avg_reward

def get_rollout_strategy(evaluator: PromptEvaluator, config: SearchConfig):
    if config.rollout_idx == 0:
        return ClassicPathRollout(evaluator, config.rollout_length)
