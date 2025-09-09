from collections import defaultdict
from typing import Any
from mcts.expand import ExpandStrategy
from mcts.choose import ChooseStrategy
from mcts.rollout import RolloutStrategy
from mcts.select import UCTStrategy
from logger import logger
from mcts.node import Node
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        self.uct_value:dict[Node, float] = dict()
        self.select_strategy = select_strategy
        self.expand_strategy = expand_strategy
        self.rollout_strategy = rollout_strategy
        self.choose_strategy = choose_strategy
        self.lock = Lock()

    def _select(self, node: Node) -> list[Node]:
        """
        自顶向下走到一个可扩展的位置：
        - 若 node 不在 self.children 中，返回路径（首次到达叶子，等待 expand）
        - 否则，使用 UCT 继续向下选择
        """
        path: list['Node'] = []
        while True:
            path.append(node)
            children_count = len(self.children.get(node, []))
            if children_count == 0 or node.is_terminal():
                return path
            node, uct_value = self._uct_select(node)
            self.uct_value[node] = uct_value

    def _uct_select(self, node: Node) -> Node:
        return self.select_strategy.select(node, self)

    def _expand(self, node: Node, expand_width=1) -> Node:
        """
        一次扩展多个 child，并返回列表。
        """
        children = self.expand_strategy.expand(node, self, expand_width)
        if not children:
            return []

        if node not in self.children:
            self.children[node] = []

        for child in children:
            if child not in self.children[node]:
                self.children[node].append(child)

        return children

    def _rollout(self, node: Node, length:int):
        return self.rollout_strategy.rollout(node, length)

    def _backpropagate(self, path:list[Node], reward):
        with self.lock:
            for node in reversed(path):
                self.N[node] += 1
                self.Q[node] = node.q_value(self.Q[node], reward)

    def do_iter(self, node: Node, expand_width:int, rollout_length:int):
        logger.info("--------------Start Iteration----------------")
        logger.info("Step 1: Performing Select")
        path = self._select(node)
        leaf = path[-1]
        logger.info(f"Selected leaf node type: {leaf.type}")
        logger.info("Step 2: Performing Expand")
        children = self._expand(leaf, expand_width)
        rollout_targets = children if children else [leaf]
        rollout_path = path.copy()

        results = []
        # Step 3: 并行 rollout
        with ThreadPoolExecutor(max_workers=len(rollout_targets)) as executor:
            future_to_child = {
                executor.submit(self._rollout, child, rollout_length): child
                for child in rollout_targets
            }
            for future in as_completed(future_to_child):
                child = future_to_child[future]
                try:
                    reward = future.result()
                    results.append((child, reward))
                    logger.info(f"Rollout target {child} score: {reward:.2f}")
                except Exception as e:
                    logger.error(f"⚠️ Error during rollout of {child}: {e}")

        # Step 4: 串行回传，确保 lock
        for child, reward in results:
            self._backpropagate(rollout_path + [child], reward)

        logger.info("---------------End Iteration------------------")

    def choose(self, root: Node) -> Node:
        return self.choose_strategy.choose(root, self)
        