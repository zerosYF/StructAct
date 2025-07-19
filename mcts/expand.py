from abc import ABC, abstractmethod
from typing import List
from mcts.node import Node
from search.config import SearchConfig
import random
import numpy as np

class ExpandStrategy(ABC):
    @abstractmethod
    def expand(self, node: Node, mcts, max_expand: int = None) -> List[Node]:
        pass

class DefaultExpandStrategy(ExpandStrategy):
    def expand(self, node: Node, mcts, max_expand: int = None) -> List[Node]:
        if node not in mcts.children:
            mcts.children[node] = []
            mcts.untried_actions[node] = node.get_untried_actions()

        actions = mcts.untried_actions[node]
        if not actions:
            return []

        k = len(actions) if max_expand is None else min(max_expand, len(actions))
    
        # 🚀 使用带 usage_count 的 softmax 随机选取 k 个动作
        selected_actions = self._weighted_random_choice(actions, k)

        # 从 untried_actions 中移除被选的动作
        for action in selected_actions:
            mcts.untried_actions[node].remove(action)

        children = []
        for action in selected_actions:
            child = node.take_action(action)
            mcts.children[node].append(child)
            mcts.untried_actions[child] = child.get_untried_actions()
            children.append(child)

        return children
    
    def _weighted_random_choice(actions: list, k: int, temperature: float = 1.0):
        """根据 usage_count 做 softmax 加权随机选择"""
        usage_counts = np.array([a.usage_count for a in actions])
        # usage 越大，概率越低；加1防止除0
        logits = -usage_counts / temperature
        probs = np.exp(logits)
        probs /= probs.sum()
        selected_indices = np.random.choice(len(actions), size=k, replace=False, p=probs)
        return [actions[i] for i in selected_indices]

def get_expand_strategy(config:SearchConfig):
    return DefaultExpandStrategy()