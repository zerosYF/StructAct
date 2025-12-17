from task.base_task import TaskBase
from src.action.base_action import OptimizeAction
from src.evaluator import PromptEvaluator
from src.mcts.mcts import MCTS
from src.action.prompt_node import PromptNode
from src.config import SearchConfig
from typing import List, Set
from visualizer import MCTSVisualizer
from src.mcts.expand import get_expand_strategy
from src.mcts.rollout import get_rollout_strategy
from src.mcts.choose import get_choose_strategy
from src.action.strategy_actions import define_full_actions
from search.search import SearchController
from src.pool.sample_pools import ContinuousSamplePool
from logger import logger
import os
import json

class DualSearchController(SearchController):
    def __init__(self, 
                 evaluator: PromptEvaluator, 
                 config: SearchConfig, 
                 task: TaskBase):
        super().__init__(evaluator, config, task)
        self.actions: Set[OptimizeAction] = define_full_actions(task)

    def search(self):
        init_prompt = self.task.origin_prompt
        optimized_prompt = self._mcts_workflow(init_prompt)
        return "", optimized_prompt
    
    def _mcts_workflow(self, init_prompt: str) -> str:
        if self.task.config.use_pool:
            self.pool = ContinuousSamplePool(
                max_size=1000,
                high_reward_threshold=self.config.high_reward_threshold,
                low_reward_threshold=self.config.low_reward_threshold,
                var_unstable=self.config.var_unstable,
                informative_threshold=self.config.informative_threshold,
                )
            self.pool.initialize(self.task.get_train_mcts(), self.evaluator, init_prompt)
        else:
            self.pool = None

        PromptNode.reset_id()
        root_node = PromptNode(
                action_set=self.actions,
                action_seq=[],
                trajectory_prompts=[],
                prompt=init_prompt,
                evaluator=self.evaluator,
                depth=0,
                sample_pool=self.pool
            )
        
        visualizer = MCTSVisualizer(root_node)
        visualizer.start()

        mcts = MCTS(
            iter_num=self.config.mcts_iter_num_max,
            max_depth=self.config.max_depth_threshold,
            min_depth=self.config.min_depth_threshold,
            expand_width=self.config.width_threshold,
            rollout_length=self.config.rollout_threshold,
            exploration_weight=self.config.exploration_weight,

            expand_strategy=get_expand_strategy(self.config),
            rollout_strategy=get_rollout_strategy(self.config),
            choose_strategy=get_choose_strategy(self.config)
        )
        mcts.min_reward_threshold = root_node.reward_value
        mcts.increase_threshold(root_node.reward_value)
        for iter_id in range(self.config.mcts_iter_num_max):
            mcts.do_iter(root_node, iter_id)

        best_node: PromptNode = mcts.choose(root_node)
        logger.info("🏁 Search completed. Selected best action sequence:")
        for i, action in enumerate(best_node.action_seq):
            logger.info(f"  Step {i+1}: {action.name}")
        best_prompt = best_node.current_prompt

        result_dict = mcts.serialize(root_node)
        os.makedirs("logs", exist_ok=True)
        file_name = f"logs/{self.task.name}_dual_mcts_full_tree_without_pool.json"
        if self.pool:
            file_name = f"logs/{self.task.name}_dual_mcts_full_tree.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Full MCTS tree has been saved to logs/{file_name}")

        return best_prompt