from collections import defaultdict
from typing import Any
from mcts.expand import ExpandStrategy
from mcts.choose import ChooseStrategy
from mcts.rollout import RolloutStrategy
from mcts.select import UCTStrategy
from logger import logger
from mcts.node import Node
import concurrent.futures
from threading import Lock

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
        self.lock = Lock()

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
        with self.lock:
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

        logger.info(f"Step 3: Running Rollouts for {len(rollout_targets)} children (parallel)")
        results = []

        def rollout_and_backprop(child_node):
            rollout_path = path.copy() + [child_node]
            reward = self._rollout(child_node)
            self._backpropagate(rollout_path, reward)
            logger.info(f"Child node {child_node} rollout score: {reward:.2f}")
            return (rollout_path, reward)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(rollout_targets)) as executor:
            futures = [executor.submit(rollout_and_backprop, child) for child in rollout_targets]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"⚠️ Error during rollout: {e}")

        logger.info("---------------End Iteration------------------")

    def choose(self, root: Node) -> Node:
        return self.choose_strategy.choose(root, self)
        