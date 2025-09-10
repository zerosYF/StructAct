from abc import ABC, abstractmethod
from search.config import SearchConfig
from search.evaluator import PromptEvaluator
from mcts.node import Node, Step
import numpy as np
import random
from logger import logger

class RolloutStrategy(ABC):
    @abstractmethod
    def rollout(self, node, rollout_width:int, mcts) -> float:
        """Perform a rollout starting from the given node and return the final reward"""
        pass

class ClassicPathRollout(RolloutStrategy):
    def rollout(self, node: Node, rollout_width:int, mcts):
        current: Node = node
        steps = 0

        while True:
            if mcts.should_early_stop(current):
                current.is_terminal = mcts.is_terminal_node(current)
                mcts.increase_threshold(current.reward_value)
                break

            mcts.increase_threshold(current.reward_value)
            steps += 1

            if mcts.is_terminal_node(current):
                break

            if current.is_leaf():
                mcts.expand(current, rollout_width)

            if len(current.children) != 0:
                current = max(current.children, key=lambda child: child.reward_value)
            else:
                current.is_terminal = True
                break

        return current


def get_rollout_strategy(config: SearchConfig):
    return ClassicPathRollout()
