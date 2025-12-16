from abc import ABC, abstractmethod
from enum import Enum
from micronet.parameters import ParamBundle
import itertools
import math
class Step(Enum):
    Expand = 0,
    Rollout = 1,

class Node(ABC):
    type:str = None
    id_iter = itertools.count()
    def __init__(self, 
                 depth:int,
                 Q:float=0.0, 
                 N:int=0, 
                 uct_value:float=0.0, 
                 parent=None, 
                 ):
        
        self.id = next(Node.id_iter)
        
        self.Q = Q  # Total reward of the node
        self.N = N    # Visit count of the node
        self.uct_value = uct_value # UCT value of the node
        self.depth = depth
        self.parent:Node = parent
        self.children:list[Node] = []
        self.cum_rewards:list[float] = []

        self.is_terminal:bool = False
    
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
    
    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def is_leaf(self):
        return len(self.children) == 0
    
    def update(self, reward):
        self.Q = self.q_value(self.Q, reward)
        self.N += 1
    
    def average_q(self):
        return self.Q / self.N if self.N > 0 else 0.0
    
    def get_exploration_weight(self, exploration_weight=1.41):
        return exploration_weight

    def compute_uct(self, exploration_weight):
        if self.parent is None:
            log_N_parent = 0
        else:
            log_N_parent = math.log(self.parent.N + 1)
        exploitation = self.average_q()
        exploration = exploration_weight * math.sqrt(log_N_parent / (self.N + 1e6))
        self.uct_value = exploitation + exploration
        return self.uct_value

    def find_best_node(self, exploration_weight=1.41):
        return max(self.children, key=lambda child: child.compute_uct(self.get_exploration_weight(exploration_weight)))

    @abstractmethod
    def reward(self):
        pass   
    @abstractmethod
    def take_action(self, params_bundle:ParamBundle, step_type:Step=Step.Expand):
        pass
    @abstractmethod
    def q_value(self, last_q, rollout_reward):
        pass
    @abstractmethod
    def serialize(self):
        pass