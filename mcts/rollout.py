from abc import ABC, abstractmethod
from search.config import SearchConfig
from concurrent.futures import ThreadPoolExecutor
from search.evaluator import PromptEvaluator
from mcts.node import Node
import numpy as np
import random
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


class MultiPathRollout(RolloutStrategy):
    def __init__(
        self,
        evaluator: PromptEvaluator,
        rollout_depth: int = 5,
        num_paths: int = 5,
        early_stop_rounds: int = 3,
        early_stop_delta: float = 0.01,
    ):
        self.rollout_depth = rollout_depth
        self.num_paths = num_paths
        self.evaluator = evaluator
        self.early_stop_rounds = early_stop_rounds
        self.early_stop_delta = early_stop_delta

    def rollout(self, node: Node) -> float:
        all_rewards = []
        avg_rewards_history = []

        for path_idx in range(self.num_paths):
            current:Node = node.clone_node()
            depth = 0
            path_rewards = []

            while depth < self.rollout_depth:
                actions = current.get_possible_actions()
                if not actions:
                    break
                action = random.choice(actions)
                current = current.take_action(action)
                reward = current.reward()
                path_rewards.append(reward)

                logger.info(f"[Rollout-{path_idx}] Step {depth+1}, Action: {getattr(action, 'name', 'Unknown')}, Reward: {reward:.4f}")
                depth += 1

            all_rewards.extend(path_rewards)

            avg_reward_now = np.mean(all_rewards) if all_rewards else 0.0
            avg_rewards_history.append(avg_reward_now)

            logger.info(f"[Rollout-{path_idx}] Current average reward: {avg_reward_now:.4f}")

            #  check early stop
            if len(avg_rewards_history) >= self.early_stop_rounds:
                recent = avg_rewards_history[-self.early_stop_rounds:]
                if max(recent) - min(recent) < self.early_stop_delta:
                    logger.info(f"🛑 Early stopping triggered at path {path_idx+1}. Avg reward: {avg_reward_now:.4f}")
                    break

        final_avg_reward = np.mean(all_rewards) if all_rewards else 0.0
        logger.info(f"✅ Multi-path rollout final average reward: {final_avg_reward:.4f}")
        return final_avg_reward

def get_rollout_strategy(evaluator: PromptEvaluator, config: SearchConfig):
    if config.rollout_idx == 0:
        return ClassicPathRollout(evaluator, config.rollout_length)
    elif config.rollout_idx == 1:
        return MultiPathRollout(evaluator, 
                                config.rollout_length, 
                                config.rollout_width, 
                                config.rollout_early_stop_rounds, 
                                config.rollout_early_stop_delta)
