from abc import ABC, abstractmethod
from search.config import SearchConfig
from search.evaluator import PromptEvaluator
from mcts.node import Node, Step
import numpy as np
import random
from logger import logger

class RolloutStrategy(ABC):
    @abstractmethod
    def rollout(self, node, rollout_length:int) -> float:
        """Perform a rollout starting from the given node and return the final reward"""
        pass

class ClassicPathRollout(RolloutStrategy):
    def __init__(
        self,
        evaluator: PromptEvaluator,
        early_stop_rounds: int = 3,
        early_stop_delta: float = 0.001,
    ):
        self.evaluator = evaluator
        self.early_stop_rounds = early_stop_rounds
        self.early_stop_delta = early_stop_delta

    def rollout(self, node: Node, rollout_length: int) -> float:
        current: Node = node.clone_node()
        depth = 0

        final_rewards = []
        avg_rewards_history = []

        while depth < rollout_length and not current.is_terminal():
            actions = current.get_possible_actions()
            if not actions:
                break

            action = random.choice(actions)
            current = current.take_action(action, Step.Rollout)
            depth += 1

            logger.info(f"[Rollout] Step {depth}, Action: {getattr(action, 'name', 'Unknown')}")

            # 计算 reward 并记录历史
            reward_now = current.reward()
            final_rewards.append(reward_now)

            avg_reward_now = np.mean(final_rewards)
            avg_rewards_history.append(avg_reward_now)
            logger.info(f"[Rollout] Current average reward: {avg_reward_now:.4f}")

            # 早停条件
            if len(avg_rewards_history) >= self.early_stop_rounds:
                recent = avg_rewards_history[-self.early_stop_rounds:]
                if max(recent) - min(recent) < self.early_stop_delta:
                    logger.info(f"🛑 Early stopping triggered at step {depth}. Avg reward: {avg_reward_now:.4f}")
                    break

        # 最终结果
        final_avg_reward = np.mean(final_rewards) if final_rewards else 0.0
        logger.info(f"✅ Single-path rollout final average reward: {final_avg_reward:.4f}")
        return final_avg_reward


def get_rollout_strategy(evaluator: PromptEvaluator, config: SearchConfig):
    return ClassicPathRollout(evaluator, config.rollout_early_stop_rounds, config.rollout_early_stop_delta)
