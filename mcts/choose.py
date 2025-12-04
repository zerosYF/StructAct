from abc import ABC, abstractmethod
from mcts.node import Node
from search.config import SearchConfig
import math
from logger import logger

class ChooseStrategy(ABC):
    @abstractmethod
    def choose(self, root):
        """Given the root node and the MCTS tree, return the best action sequence"""
        pass

class MaxLeafQnStrategy(ChooseStrategy):
    def choose(self, root: Node):
        best_node = None
        best_score = float("-inf")
        visited = set()
        stack = [root]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            if node.is_leaf():
                n = node.N
                if n > 0:
                    score = node.Q / n
                    if score > best_score:
                        best_score = score
                        best_node = node
            stack.extend(node.children)

        if best_node:
            return best_node
        else:
            logger.warning("⚠️ MaxLeafQnStrategy did not find a valid action sequence")
            return []


class MaxQNStrategy(ChooseStrategy):
    def choose(self, root: Node):
        best_node = None
        best_score = float("-inf")
        visited = set()
        stack = [root]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            n = node.N
            if n > 0:
                score = node.Q / n
                if score > best_score:
                    best_score = score
                    best_node = node

            stack.extend(node.children)

        if best_node:
            return best_node
        else:
            logger.warning("⚠️ MaxLeafQnStrategy did not find a valid action sequence")
            return []

class MaxPathBestNodeStrategy(ChooseStrategy):
    def choose(self, root: Node):
        best_path = None
        best_path_score = float("-inf")

        # DFS stack: (current_node, current_path, accumulated_rewards)
        stack = [(root, [root], [root.reward_value])]

        while stack:
            node, path, rewards = stack.pop()

            if node.is_leaf():
                avg_reward = sum(rewards) / len(rewards)
                if avg_reward > best_path_score:
                    best_path_score = avg_reward
                    best_path = (path, rewards)
            else:
                for child in node.children:
                    stack.append((child, path + [child], rewards + [child.reward_value]))

        if best_path:
            path, rewards = best_path
            max_idx = max(range(len(rewards)), key=lambda i: rewards[i])
            return path[max_idx]
        else:
            logger.warning("⚠️ MaxPathBestNodeStrategy did not find a valid action sequence")
            return []

def get_choose_strategy(config: SearchConfig):
    return MaxPathBestNodeStrategy()
