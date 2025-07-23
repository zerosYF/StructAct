from collections import defaultdict
from typing import Any
from mcts.expand import ExpandStrategy
from mcts.choose import ChooseStrategy
from mcts.rollout import RolloutStrategy
from mcts.select import UCTStrategy
from logger import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from mcts.node import Node

class MCTS:
    def __init__(self, 
                 select_strategy: UCTStrategy = None,
                 expand_strategy: ExpandStrategy = None,
                 rollout_strategy: RolloutStrategy = None,
                 choose_strategy: ChooseStrategy = None, 
                ):
        self.Q = defaultdict(int)  # Historical reward of paths
        self.N = defaultdict(int)  # Number of times nodes have been explored
        self.children: dict[Node, list[Node]] = dict()  # Tree structure storage
        self.untried_actions: dict[Node, list[Any]] = dict()
        self.select_strategy = select_strategy
        self.expand_strategy = expand_strategy
        self.rollout_strategy = rollout_strategy
        self.choose_strategy = choose_strategy

    def _select(self, node: Node, max_width: int) -> list[Node]:
        path = []
        while True:
            path.append(node)
            if node not in self.children:
                return path
            unexplored = self.untried_actions.get(node, [])
            children_count = len(self.children.get(node, []))

            # Return current node only if the number of children is less than max_width and there are untried actions; otherwise continue down the tree
            if unexplored and children_count < max_width:
                return path
            node = self._uct_select(node)

    def _uct_select(self, node: Node) -> Node:
        return self.select_strategy.select(node, self)

    def _expand(self, node: Node, max_expand: int = None) -> list[Node]:
        return self.expand_strategy.expand(node, self, max_expand)

    def _rollout(self, node: Node):
        return self.rollout_strategy.rollout(node)

    def _backpropagate(self, path:list[Node], reward):
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] = node.q_value(self.Q[node], reward)

    def do_iter(self, node: Node, width: int = 1, expand_num: int = 1):
        logger.info("--------------Start Iteration----------------")
        logger.info("Step 1: Performing Select")
        path = self._select(node, max_width=width)
        leaf = path[-1]
        logger.info(f"Selected leaf node type: {leaf.type}")
        logger.info("Step 2: Performing Expand")
        children = self._expand(leaf, expand_num)
        if not children:
            rollout_targets = [leaf]
        else:
            rollout_targets = children

        logger.info("Step 3: Running Rollouts")
        results = []
        for child in rollout_targets:
            rollout_path = path.copy()
            rollout_path.append(child)
            reward = self._rollout(child)
            logger.info(f"Child node {child} rollout score: {reward:.2f}")
            results.append((rollout_path, reward))

        logger.info("Step 4: Backpropagating reward for each child node")
        for rollout_path, reward in results:
            self._backpropagate(rollout_path, reward)

        logger.info("---------------End Iteration------------------")

    def choose(self, root: Node) -> Node:
        return self.choose_strategy.choose(root, self)
        