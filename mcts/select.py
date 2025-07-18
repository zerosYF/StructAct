from search.config import SearchConfig  
from abc import ABC, abstractmethod
import math

class UCTStrategy(ABC):
    @abstractmethod
    def select(self, node, mcts):
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
        
        return max(children, key=uct_value)

class PriorUCTStrategy(UCTStrategy):
    def __init__(self, policy_controller, exploration_weight=1.0):
        self.policy_controller = policy_controller
        self.exploration_weight = exploration_weight

    def select(self, node, mcts):
        children = mcts.children.get(node, [])
        if not children:
            return None

        # 当前节点访问次数
        log_N_parent = math.log(mcts.N.get(node, 1) + 1e-6)

        prior_probs = 1.0

        def uct_value(child_node):
            try:
                action = child_node.action_seq[-1]
                pi = prior_probs
            except (ValueError, IndexError):
                pi = 1.0 

            q = mcts.Q.get(child_node, 0)
            n = mcts.N.get(child_node, 0)
            return q / (n + 1e-6) + self.exploration_weight * pi * math.sqrt(log_N_parent / (n + 1e-6))

        return max(children, key=uct_value)

def get_select_strategy(config:SearchConfig, policy=None):
        if config.uct_idx == 0:
            return ClassicUCTStrategy(config.exploration_weight)
        else:
            return PriorUCTStrategy(policy, config.exploration_weight)