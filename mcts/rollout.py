from abc import ABC, abstractmethod
from search.config import SearchConfig
from search.evaluator import PromptEvaluator
from mcts.node import Node, Step
import concurrent.futures
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
            current = current.take_action(action, Step.Rollout)

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

    def _rollout_path(self, node: Node, path_idx: int) -> float:
        current: Node = node.clone_node()
        depth = 0

        while depth < self.rollout_depth:
            actions = current.get_possible_actions()
            if not actions:
                break
            action = random.choice(actions)
            current = current.take_action(action, Step.Rollout)
            depth += 1
            logger.info(f"[Rollout-{path_idx}] Step {depth}, Action: {getattr(action, 'name', 'Unknown')}")

        final_reward = current.reward()
        logger.info(f"[Rollout-{path_idx}] Final reward: {final_reward:.4f}")
        return final_reward

    def rollout(self, node: Node) -> float:
        final_rewards = []
        avg_rewards_history = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_paths) as executor:
            future_to_path = {
                executor.submit(self._rollout_path, node, path_idx): path_idx
                for path_idx in range(self.num_paths)
            }

            for future in concurrent.futures.as_completed(future_to_path):
                path_idx = future_to_path[future]
                try:
                    final_reward = future.result()
                    final_rewards.append(final_reward)

                    avg_reward_now = np.mean(final_rewards)
                    avg_rewards_history.append(avg_reward_now)

                    logger.info(f"[Rollout-{path_idx}] Current average reward: {avg_reward_now:.4f}")

                    # Early stopping condition
                    if len(avg_rewards_history) >= self.early_stop_rounds:
                        recent = avg_rewards_history[-self.early_stop_rounds:]
                        if max(recent) - min(recent) < self.early_stop_delta:
                            logger.info(f"🛑 Early stopping triggered at path {path_idx + 1}. Avg reward: {avg_reward_now:.4f}")
                            break

                except Exception as e:
                    logger.error(f"Error in rollout path {path_idx}: {e}")

        final_avg_reward = np.mean(final_rewards) if final_rewards else 0.0
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
