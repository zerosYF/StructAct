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
                 select_strategy:UCTStrategy=None,
                 expand_strategy:ExpandStrategy=None,
                 rollout_strategy:RolloutStrategy=None,
                 choose_strategy:ChooseStrategy=None, 
                ):
        self.Q = defaultdict(int) #路径历史奖励
        self.N = defaultdict(int) #节点探索次数
        self.children:dict[Node, list[Node]] = dict() #树结构存储
        self.untried_actions:dict[Node, list[Any]] = dict() 
        self.select_strategy = select_strategy
        self.expand_strategy = expand_strategy
        self.rollout_strategy = rollout_strategy
        self.choose_strategy = choose_strategy
    def _select(self, node:Node, max_width:int):
        path = []
        while True:
            path.append(node)
            if node not in self.children:
                return path
            unexplored = self.untried_actions.get(node, [])
            children_count = len(self.children.get(node, []))

            # 只有子节点没达到上限且有未尝试动作才返回当前节点，否则继续下探
            if unexplored and children_count < max_width:
                return path
            node = self._uct_select(node)
    
    def _uct_select(self, node:Node):
        return self.select_strategy.select(node, self)
    
    def _expand(self, node: Node, max_expand: int = None):
        return self.expand_strategy.expand(node, self, max_expand)

    def _rollout(self, node:Node):
        return self.rollout_strategy.rollout(node)
    
    def _backpropagate(self, path, reward):
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
    
    def do_iter(self, node:Node, width:int=1, expand_num:int=1, rollout_parallel=False):
        logger.info("--------------Start Iteration----------------")
        logger.info("Step 1: 执行Select")
        path = self._select(node, max_width=width)
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
       
        