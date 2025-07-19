from abc import ABC, abstractmethod
from mcts.node import Node
from search.config import SearchConfig
import math

class ChooseStrategy(ABC):
    @abstractmethod
    def choose(self, root, mcts):
        """Given the root node and the MCTS tree, return the best action sequence"""
        pass

class MaxLeafQnStrategy(ChooseStrategy):
    def choose(self, root: Node, mcts):
        best_node = None
        best_score = float("-inf")
        visited = set()
        stack = [root]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            if node.is_terminal() or node not in mcts.children:
                n = mcts.N.get(node, 0)
                if n > 0:
                    score = mcts.Q[node] / n
                    if score > best_score:
                        best_score = score
                        best_node = node
            stack.extend(mcts.children.get(node, []))

        if best_node:
            return best_node
        else:
            mcts.logger.warning("⚠️ MaxLeafQnStrategy did not find a valid action sequence")
            return []

class MaxPathAvgQnStrategy(ChooseStrategy):
    def choose(self, root: Node, mcts):
        best_path = []
        best_score = float("-inf")
        stack = [(root, [])]
        visited = set()

        while stack:
            node, path = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            path = path + [node]
            if node.is_terminal() or node not in mcts.children:
                scores = [mcts.Q[n] / mcts.N[n] for n in path if mcts.N[n] > 0]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_path = path
            else:
                for child in mcts.children.get(node, []):
                    stack.append((child, path))

        return best_path[-1]

class WeightedPathQnStrategy(ChooseStrategy):
    def choose(self, root: Node, mcts):
        best_path = []
        best_score = float("-inf")
        stack = [(root, [])]
        visited = set()

        while stack:
            node, path = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            new_path = path + [node]
            if node.is_terminal() or node not in mcts.children:
                weights = [mcts.N[n] for n in new_path if mcts.N[n] > 0]
                scores = [mcts.Q[n] / mcts.N[n] for n in new_path if mcts.N[n] > 0]
                if weights:
                    total_weight = sum(weights)
                    weighted_score = sum(w * s for w, s in zip(weights, scores)) / total_weight
                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_path = new_path
            else:
                for child in mcts.children.get(node, []):
                    stack.append((child, new_path))

        return best_path[-1]

class SoftmaxPathQnStrategy(ChooseStrategy):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def choose(self, root: Node, mcts):
        best_path = []
        best_score = float("-inf")
        stack = [(root, [])]
        visited = set()

        while stack:
            node, path = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            new_path = path + [node]
            if node.is_terminal() or node not in mcts.children:
                qn_values = [mcts.Q[n] / mcts.N[n] for n in new_path if mcts.N[n] > 0]
                if qn_values:
                    exp_vals = [math.exp(self.alpha * s) for s in qn_values]
                    total = sum(exp_vals)
                    softmax_weights = [e / total for e in exp_vals]
                    weighted_score = sum(w * s for w, s in zip(softmax_weights, qn_values))
                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_path = new_path
            else:
                for child in mcts.children.get(node, []):
                    stack.append((child, new_path))

        return best_path[-1]

class MaxQNStrategy(ChooseStrategy):
    def choose(self, root: Node, mcts):
        best_node = None
        best_qn = float("-inf")

        # Choose the node with the highest Q/N value
        for node, q in mcts.Q.items():
            if node == root:  # Exclude the root node
                continue
            n = mcts.N.get(node, 0)
            if n > 0:  # Only nodes with visit count greater than 0 participate in selection
                qn = q / n  # Consider both Q and N
                if qn > best_qn:
                    best_qn = qn
                    best_node = node

        if best_node is not None:
            return best_node
        else:
            mcts.logger.warning("⚠️ MaxQNStrategy did not find a suitable node, returning the root node")
            return root  # If no suitable node is found, consider returning the root node or another fallback strategy

class MaxFinalQnOnLongestPathStrategy(ChooseStrategy):
    def choose(self, root: Node, mcts):
        stack = [(root, [])]
        best_node = None
        best_qn = float("-inf")

        while stack:
            node, path = stack.pop()
            new_path = path + [node]
            if node.is_terminal() or node not in mcts.children:
                last = new_path[-1]
                if mcts.N.get(last, 0) > 0:
                    qn = mcts.Q[last] / mcts.N[last]
                    if qn > best_qn:
                        best_qn = qn
                        best_node = last
            else:
                for child in mcts.children[node]:
                    stack.append((child, new_path))

        return best_node or root

def get_choose_strategy(config: SearchConfig):
    if config.choose_idx == 0:
        return MaxQNStrategy()
    elif config.choose_idx == 1:
        return MaxPathAvgQnStrategy()
    else:
        return MaxFinalQnOnLongestPathStrategy()