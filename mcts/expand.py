from abc import ABC, abstractmethod
from typing import List
from mcts.node import Node, Step
from search.config import SearchConfig
import threading
import concurrent.futures
import numpy as np
from logger import logger

class ExpandStrategy(ABC):
    @abstractmethod
    def expand(self, node: Node, mcts, max_expand: int = None) -> List[Node]:
        """Expand the given node in the MCTS tree, returning a list of child nodes."""
        pass

class DefaultExpandStrategy(ExpandStrategy):
    def __init__(self, max_expand: int = 1):
        self.lock = threading.Lock()
        self.max_expand = max_expand

    def expand(self, node: Node, mcts) -> List[Node]:
        if node not in mcts.children:
            mcts.children[node] = []
            mcts.untried_actions[node] = node.get_untried_actions()

        actions = list(mcts.untried_actions[node])
        if not actions:
            return []

        k = len(actions) if self.max_expand is None else min(self.max_expand, len(actions))

        # 🚀 Use softmax weighted random selection based on usage_count to pick k actions
        selected_actions = self._weighted_random_choice(actions, k)

        # Remove selected actions from untried_actions
        for action in selected_actions:
            mcts.untried_actions[node].remove(action)

        children = []
        if k == 0:
            return children
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=k) as executor:
            futures = {
                executor.submit(self._expand_action_threadsafe, node, action, mcts): action
                for action in selected_actions
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    child = future.result()
                    if child is not None:
                        children.append(child)
                except Exception as e:
                    action = futures[future]
                    logger.error(f"Error expanding action {getattr(action, 'name', 'Unknown')}: {e}")
        return children
    
    def _expand_action_threadsafe(self, node: Node, action, mcts) -> Node:
        try:
            child: Node = node.take_action(action, Step.Expand)
            with self.lock:
                mcts.children[node].append(child)
                mcts.untried_actions[child] = child.get_untried_actions()
            return child
        except Exception as e:
            logger.error(f"Exception in take_action: {e}")
            return None
    
    def _weighted_random_choice(self, actions: list, k: int, temperature: float = 1.0):
        """Softmax weighted random selection based on usage_count"""
        usage_counts = np.array([a.usage_count for a in actions])
        # Higher usage_count means lower probability; add 1 to avoid division by zero
        logits = -usage_counts / temperature
        probs = np.exp(logits)
        probs /= probs.sum()
        selected_indices = np.random.choice(len(actions), size=k, replace=False, p=probs)
        return [actions[i] for i in selected_indices]

def get_expand_strategy(max_expand: int = None) -> ExpandStrategy:
    return DefaultExpandStrategy(max_expand)