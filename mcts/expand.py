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

    def expand(self, node: Node, mcts, expand_width: int) -> list[Node]:
        import concurrent.futures
        if node.is_terminal():
            return []

        if node not in mcts.children:
            mcts.children[node] = []

        actions = list(node.get_untried_actions())
        if not actions:
            return []

        selected_actions = []
        # 明确循环 expand_width 次，每次按 usage_count 采样 1 个（允许重复）
        for _ in range(expand_width):
            action = self._weighted_random_choice(actions)  # 返回单个 action
            if action is not None:
                selected_actions.append(action)

        if not selected_actions:
            return []

        children: list[Node] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_actions)) as executor:
            futures = {executor.submit(self._expand_action_threadsafe, node, act, mcts): act
                    for act in selected_actions}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    child = fut.result()
                    if child is not None:
                        children.append(child)
                except Exception as e:
                    action = futures[fut]
                    logger.error(f"Error expanding action {getattr(action, 'name', 'Unknown')}: {e}")

        return children
    
    def _expand_action_threadsafe(self, node: Node, action, mcts) -> Node:
        try:
            child: Node = node.take_action(action, Step.Expand)
            return child
        except Exception as e:
            logger.error(f"Exception in take_action: {e}")
            return None
    
    def _weighted_random_choice(self, actions: list, temperature: float = 1.0):
        """Softmax weighted random selection of a single action based on failure_counter and usage_count"""
        if not actions:
            return None

        failure_counts = np.array([a.sample_failure_counter for a in actions])
        usage_counts = np.array([a.usage_count for a in actions])

        logits = - (failure_counts + usage_counts) / temperature 
        logits /= temperature  

        probs = np.exp(logits)
        probs /= probs.sum() 

        selected_index = np.random.choice(len(actions), p=probs)
        
        return actions[selected_index]

def get_expand_strategy(config:SearchConfig) -> ExpandStrategy:
    return DefaultExpandStrategy(config)