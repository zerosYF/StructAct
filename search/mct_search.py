from task.base_task import TaskBase
from program.base_action import OptimizeAction
from search.evaluator import PromptEvaluator
from mcts.mcts import MCTS
from search.prompt_node import PromptNode
from search.config import SearchConfig
from typing import List, Set
from visualizer import Visualizer
from mcts.select import get_select_strategy
from mcts.expand import get_expand_strategy
from mcts.rollout import get_rollout_strategy
from mcts.choose import get_choose_strategy
from program.strategy_actions import define_full_actions
from search.search import SearchController
from program.sample_pools import DynamicSamplePool
from logger import logger
import os
import json

class MCTSearchController(SearchController):
    def __init__(self, 
                 evaluator: PromptEvaluator, 
                 config: SearchConfig, 
                 task: TaskBase):
        super().__init__(evaluator, config, task)
        self.actions: Set[OptimizeAction] = define_full_actions(task)
        self.pool: DynamicSamplePool = DynamicSamplePool(max_size=1000, low=0.5, high=0.9)

    def search(self):
        init_prompt = self.task.origin_prompt

        Visualizer.start(title=self.task.name)
        optimized_prompt = self._mcts_workflow(init_prompt)
        return "", optimized_prompt
    
    def _mcts_workflow(self, init_prompt: str):
        self.pool.initialize(self.task.get_train_mcts(), self.evaluator, init_prompt)
        root_node = PromptNode(
                action_set=self.actions,
                action_seq=[],
                trajectory_prompts=[],
                prompt=init_prompt,
                evaluator=self.evaluator,
                depth=0,
                max_depth=self.config.depth_threshold,
                sample_pool=self.pool
            )

        mcts_iters = self.config.mcts_iter_num_max
        rollout_len = self.config.rollout_length_max
        expand_width = self.config.width_threshold
        print(f"MCTS Iterations: {mcts_iters}, Rollout Length: {rollout_len}, Expand Width: {expand_width}")

        mcts = MCTS(
            select_strategy=get_select_strategy(self.config),
            expand_strategy=get_expand_strategy(self.config),
            rollout_strategy=get_rollout_strategy(self.evaluator, self.config),
            choose_strategy=get_choose_strategy(self.config)
        )

        Visualizer.set_mcts(mcts, root_node)

        for iter_id in range(mcts_iters):
            mcts.do_iter(root_node, 
                              expand_width=expand_width, 
                              rollout_length=rollout_len)
            if iter_id % 5 == 0 or iter_id == mcts_iters - 1:
                logger.info(f"  Total expanded nodes: {len(mcts.N)}")

        best_node: PromptNode = mcts.choose(root_node)
        logger.info("🏁 Search completed. Selected best action sequence:")
        for i, action in enumerate(best_node.action_seq):
            logger.info(f"  Step {i+1}: {action.name}")
        best_prompt = best_node.current_prompt

        # ==== 存储完整树 ====
        result_dict = {
            "config": {
                "mcts_iters": mcts_iters,
                "rollout_length": rollout_len,
                "depth_threshold": self.config.depth_threshold,
                "width_threshold": self.config.width_threshold,
            },
            "search_stats": {
                "total_nodes": len(mcts.N),
                "total_Q_values": len(mcts.Q),
            },
            "best_node": {
                "action_sequence": [a.name for a in best_node.action_seq],
                "prompt": best_node.current_prompt,
                "depth": best_node.depth,
                "Q": mcts.Q.get(best_node, 0.0),
                "N": mcts.N.get(best_node, 0)
            },
            "search_tree": self._serialize_node(root_node, mcts)
        }

        os.makedirs("logs", exist_ok=True)
        with open(f"logs/{self.task.name}_mcts_full_tree.json", "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        logger.info("✅ Full MCTS tree has been saved to logs/mcts_full_tree.json")

        return best_prompt
    
    def _serialize_node(self, node, mcts, visited=None, node_id_map=None, next_id=[0]):
        """
        递归序列化节点及其子树
        """
        if visited is None:
            visited = set()
        if node_id_map is None:
            node_id_map = {}

        if node in visited:
            return None
        visited.add(node)

        # 给每个节点分配一个 ID
        if node not in node_id_map:
            node_id_map[node] = next_id[0]
            next_id[0] += 1

        node_id = node_id_map[node]

        # 获取 Q/N
        q_val = mcts.Q.get(node, 0.0)
        n_val = mcts.N.get(node, 0)

        # 当前节点字典
        node_dict = {
            "id": node_id,
            "depth": node.depth,
            "action_sequence": [a.name for a in node.action_seq],
            "prompt": node.current_prompt,
            "Q": q_val,
            "N": n_val, 
            "reward": node.reward_value,
            "children": []
        }

        # 遍历子节点
        for child in mcts.children.get(node, []):
            child_serialized = self._serialize_node(child, mcts, visited, node_id_map, next_id)
            if child_serialized:
                node_dict["children"].append(child_serialized)

        return node_dict