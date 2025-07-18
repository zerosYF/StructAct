from abc import ABC, abstractmethod
from typing import List
from mcts.node import Node
from search.config import SearchConfig

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
        children = []

        for _ in range(k):
            action = actions.pop()
            child = node.take_action(action)
            mcts.children[node].append(child)
            mcts.untried_actions[child] = child.get_untried_actions()
            children.append(child)

        return children

def get_expand_strategy(config:SearchConfig):
    return DefaultExpandStrategy()