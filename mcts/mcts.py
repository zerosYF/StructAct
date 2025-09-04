from collections import defaultdict
from typing import Any
from mcts.expand import ExpandStrategy
from mcts.choose import ChooseStrategy
from mcts.rollout import RolloutStrategy
from mcts.select import UCTStrategy
from logger import logger
from mcts.node import Node
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
        self.uct_value:dict[Node, float] = dict()
        self.select_strategy = select_strategy
        self.expand_strategy = expand_strategy
        self.rollout_strategy = rollout_strategy
        self.choose_strategy = choose_strategy
        self.lock = Lock()

    def _select(self, node: Node, max_width: int) -> list[Node]:
        """
        自顶向下走到一个可扩展的位置：
        - 若 node 不在 self.children 中，返回路径（首次到达叶子，等待 expand）
        - 若 node 的子节点数 < max_children，返回路径（允许在此处扩展一个）
        - 否则，使用 UCT 继续向下选择
        """
        path: list['Node'] = []
        while True:
            path.append(node)
            if node not in self.children:
                return path
            children_count = len(self.children.get(node, []))
            if children_count < max_width:
                return path
            node = self._uct_select(node)

    def _uct_select(self, node: Node) -> Node:
        return self.select_strategy.select(node, self)

    def _expand(self, node: Node) -> Node:
        """
        一次只扩展一个 child。
        兼容新的 expand_strategy 接口：expand(node, mcts) -> Optional[Node]
        """
        child = self.expand_strategy.expand(node, self)
        if child is None:
            return None

        # 确保 children 字典同步（若 expand_strategy 内部已维护，这里不会重复添加）
        if node not in self.children:
            self.children[node] = []
        if child not in self.children[node]:
            self.children[node].append(child)
        return child

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
        path = self._select(node, max_width=expand_width)
        leaf = path[-1]
        logger.info(f"Selected leaf node type: {leaf.type}")
        logger.info("Step 2: Performing Expand")
        child = self._expand(leaf)
        # 若无法扩展（终止状态或策略返回 None），对 leaf 自身 rollout；否则对 child rollout
        rollout_target = child if child is not None else leaf

        logger.info("Step 3: Running Rollout")
        rollout_path = path.copy()
        if child is not None:
            rollout_path.append(child)

        try:
            reward = self._rollout(rollout_target, rollout_length)
            self._backpropagate(rollout_path, reward)
            logger.info(f"Rollout target {rollout_target} score: {reward:.2f}")
        except Exception as e:
            logger.error(f"⚠️ Error during rollout: {e}")

        logger.info("---------------End Iteration------------------")

    def choose(self, root: Node) -> Node:
        return self.choose_strategy.choose(root, self)
        