from abc import ABC, abstractmethod
from typing import List
from mcts.node import Node, Step
from search.config import SearchConfig
import threading
import numpy as np
from logger import logger

class ExpandStrategy(ABC):
    @abstractmethod
    def expand(self, node: Node, mcts, expand_width) -> Node:
        """Expand the given node in the MCTS tree, returning a list of child nodes."""
        pass

class DefaultExpandStrategy(ExpandStrategy):
    def __init__(self, config:SearchConfig):
        self.lock = threading.Lock()
        self.config = config

    def expand(self, node: Node, mcts) -> Node:
        if node.is_terminal():
            return None

        if node not in mcts.children:
            mcts.children[node] = []

        actions = list(node.get_untried_actions())
        if not actions:
            return None

        # 🚀 Use softmax weighted random selection based on usage_count to pick k actions
        selected_action = self._weighted_random_choice(actions)

        try:
            child = self._expand_action_threadsafe(node, selected_action, mcts)
            return child
        except Exception as e:
            logger.error(f"Error expanding action {getattr(selected_action, 'name', 'Unknown')}: {e}")
            return None
    
    def _expand_action_threadsafe(self, node: Node, action, mcts) -> Node:
        try:
            child: Node = node.take_action(action, Step.Expand)
            with self.lock:
                mcts.children[node].append(child)
            return child
        except Exception as e:
            logger.error(f"Exception in take_action: {e}")
            return None
    
    def _weighted_random_choice(self, actions: list, temperature: float = 1.0):
        """Softmax weighted random selection of a single action based on usage_count"""
        if not actions:
            return None

        usage_counts = np.array([a.usage_count for a in actions])
        # Higher usage_count → lower probability; add 1 to avoid division by zero
        logits = -usage_counts / temperature
        probs = np.exp(logits)
        probs /= probs.sum()

        selected_index = np.random.choice(len(actions), p=probs)
        return actions[selected_index]

def get_expand_strategy(config:SearchConfig) -> ExpandStrategy:
    return DefaultExpandStrategy(config)