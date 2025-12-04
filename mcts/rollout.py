from abc import ABC, abstractmethod
from search.config import SearchConfig
from search.evaluator import PromptEvaluator
from mcts.node import Node, Step
import numpy as np
import random
from logger import logger

class RolloutStrategy(ABC):
    @abstractmethod
    def rollout(self, node, rollout_length:int, mcts) -> float:
        """Perform a rollout starting from the given node and return the final reward"""
        pass

class ClassicPathRollout(RolloutStrategy):
    def rollout(self, node: Node, rollout_length:int, mcts):
        current: Node = node
        steps = 0

        final_rewards = []
        avg_rewards_history = []

        while steps < rollout_length:
            if mcts.should_early_stop(current):
                current.is_terminal = mcts.is_terminal_node(current)
                mcts.increase_threshold(current.reward_value)
                break

            mcts.increase_threshold(current.reward_value)
            steps += 1

            if mcts.is_terminal_node(current):
                break

            if current.is_leaf():
                current = current.take_action(Step.Rollout)
                reward_now = current.reward_value
                final_rewards.append(reward_now)
                avg_reward_now = np.mean(final_rewards)
                avg_rewards_history.append(avg_reward_now)
                logger.info(f"[Rollout] Current average reward: {avg_reward_now:.4f}")
        
        final_avg_reward = np.mean(final_rewards) if final_rewards else 0.0
        logger.info(f"[Rollout] Current average reward: {avg_reward_now:.4f}")
        return final_avg_reward

class QuickRollout(RolloutStrategy):
    def rollout(self, node: Node, rollout_length:int, mcts):
        current: Node = node
        steps = 0

        final_rewards = []

        while True:
            # 1. Early stop
            if mcts.should_early_stop(current):
                current.is_terminal = mcts.is_terminal_node(current)
                mcts.increase_threshold(current.reward_value)
                final_rewards.append(current.reward_value)
                break

            # 2. Threshold update
            mcts.increase_threshold(current.reward_value)
            steps += 1

            # 3. Terminal check
            if mcts.is_terminal_node(current):
                current.is_terminal = True
                final_rewards.append(current.reward_value)
                break

            # 4. Expand if leaf
            if current.is_leaf():
                mcts._expand(current, 3)

            # 5. Move to next node (greedy by reward_value)
            if current.children:
                current = max(current.children, key=lambda child: child.reward_value)
                final_rewards.append(current.reward_value)
            else:
                current.is_terminal = True
                break

        # 7. Final average reward
        final_avg_reward = np.mean(final_rewards) if final_rewards else 0.0
        logger.info(f"[Rollout] Steps={steps}, Final avg reward={final_avg_reward:.4f}")
        return final_avg_reward


def get_rollout_strategy(config: SearchConfig):
    return ClassicPathRollout()
