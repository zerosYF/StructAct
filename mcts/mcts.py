from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any
from mcts.choose import ChooseStrategy
from mcts.choose import MaxLeafQnStrategy
from mcts.rollout import RolloutStrategy
from mcts.rollout import MultiPathRollout
from mcts.uct import UCTStrategy
from mcts.uct import ClassicUCTStrategy
from logger import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from mcts.node import Node

class MCTS(ABC):
    def __init__(self, 
                 choose_strategy:ChooseStrategy=None, 
                 rollout_strategy:RolloutStrategy=None,
                 uct_strategy:UCTStrategy=None):
        self.Q = defaultdict(int) #路径历史奖励
        self.N = defaultdict(int) #节点探索次数
        self.children:dict[Node, list[Node]] = dict() #树结构存储
        self.untried_actions:dict[Node, list[Any]] = dict() 
        self.choose_strategy = choose_strategy or MaxLeafQnStrategy()
        self.rollout_strategy = rollout_strategy or MultiPathRollout()
        self.uct_strategy = uct_strategy or ClassicUCTStrategy()
    def _select(self, node:Node):
        path = []
        while True:
            path.append(node)
            if node not in self.children:
                return path
            unexplored = self.untried_actions.get(node, [])
            if unexplored:
                return path
            node = self._uct_select(node)
    
    def _expand(self, node: Node, max_expand: int = None):
        if node not in self.children:
            self.children[node] = []
            self.untried_actions[node] = node.get_untried_actions()

        actions = self.untried_actions[node]
        if not actions:
            return None

        # 默认展开所有，或最多 max_expand 个动作
        k = len(actions) if max_expand is None else min(max_expand, len(actions))
        children = []

        for _ in range(k):
            action = actions.pop()
            child = node.take_action(action)
            self.children[node].append(child)
            self.untried_actions[child] = child.get_untried_actions()
            children.append(child)

        return children

    def _rollout(self, node:Node):
        return self.rollout_strategy.rollout(node, self)
    
    def _backpropagate(self, path, reward):
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
    
    def do_iter(self, node:Node, expand_num:int=1, rollout_parallel=False):
        logger.info("--------------Start Iteration----------------")
        logger.info("Step 1: 执行Select")
        path = self._select(node)
        leaf = path[-1]
        logger.info(f"已选叶子结点类型:{leaf.type}")
        logger.info("Step 2: 执行Expand")
        children = self._expand(leaf, expand_num)
        if not children:
            rollout_targets = [leaf]
        else:
            rollout_targets = children

        logger.info("Step 3: 并行Rollout")
        results = self._run_rollout_batch(rollout_targets, path, rollout_parallel)

        logger.info("Step 4: 回传每个子节点的 reward")
        for rollout_path, reward in results:
            self._backpropagate(rollout_path, reward)

        logger.info("---------------End Iteration------------------")
    
    def choose(self, root:Node) -> Node:
        return self.choose_strategy.choose(root, self)

    def _uct_select(self, node:Node):
        return self.uct_strategy.select(node, self)
    
    def _run_rollout_batch(self, rollout_targets, path, rollout_parallel):
        results = []

        def rollout_fn(child):
            rollout_path = path.copy()
            rollout_path.append(child)
            reward = self._rollout(child)  # 其中可调用 RNN 并训练
            return rollout_path, reward

        if rollout_parallel:
            with ThreadPoolExecutor(max_workers=len(rollout_targets)) as executor:
                futures = [executor.submit(rollout_fn, child) for child in rollout_targets]
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for child in rollout_targets:
                rollout_path, reward = rollout_fn(child)
                logger.info(f"子节点 {child} rollout 得分: {reward:.2f}")
                results.append((rollout_path, reward))

        return results
    
        