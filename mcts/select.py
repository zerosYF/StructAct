from search.config import SearchConfig  
from abc import ABC, abstractmethod
import math

class UCTStrategy(ABC):
    @abstractmethod
    def select(self, node, mcts):
        """Select a child node from the given node using UCT strategy."""
        pass

class ClassicUCTStrategy(UCTStrategy):
    def __init__(self, exploration_weight=0.1):
        self.exploration_weight = exploration_weight

    def select(self, node, mcts):
        children = mcts.children.get(node, [])
        if not children:
            return None
        log_N_parent = math.log(mcts.N[node] + 1e-6)
        
        def uct_value(n):
            q = mcts.Q[n]
            n_visits = mcts.N[n]
            return q / (n_visits + 1e-6) + self.exploration_weight * math.sqrt(log_N_parent / (n_visits + 1e-6))
        
        return max(children, key=uct_value), uct_value


def get_select_strategy(config: SearchConfig, policy=None):
    return ClassicUCTStrategy(config.exploration_weight)