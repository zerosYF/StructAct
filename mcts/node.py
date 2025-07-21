from abc import ABC, abstractmethod
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
    def take_action(self, action):
        pass
    @abstractmethod
    def clone_node(self):
        pass