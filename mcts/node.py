from abc import ABC, abstractmethod
from enum import Enum
class Step(Enum):
    Expand = 0,
    Rollout = 1,

class Node(ABC):
    type:str = None
    @abstractmethod
    def is_terminal(self):
        pass
    @abstractmethod
    def reward(self):
        pass   
    @abstractmethod
    def get_untried_actions(self):
        pass
    @abstractmethod
    def get_possible_actions(self):
        pass
    @abstractmethod
    def take_action(self, action, step_type:Step=Step.Expand):
        pass
    @abstractmethod
    def clone_node(self):
        pass
    @abstractmethod
    def q_value(self, last_q, rollout_reward):
        pass