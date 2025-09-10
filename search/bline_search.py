from task.base_task import TaskBase
from program.base_action import OptimizeAction
from search.evaluator import PromptEvaluator
from mcts.mcts import MCTS
from search.prompt_node import PromptNode
from search.config import SearchConfig
from typing import List, Set
from visualizer import MCTSVisualizer
from mcts.expand import get_expand_strategy
from mcts.rollout import get_rollout_strategy
from mcts.choose import get_choose_strategy
from program.strategy_actions import define_failure_actions
from search.search import SearchController
from logger import logger
import os
import json

class PromptAgentController(SearchController):
    def __init__(self, 
                 evaluator: PromptEvaluator, 
                 config: SearchConfig, 
                 task: TaskBase):
        super().__init__(evaluator, config, task)
        self.actions: Set[OptimizeAction] = define_failure_actions(task)

    def search(self):
        init_prompt = self.task.origin_prompt

        optimized_prompt = self._mcts_workflow(init_prompt)
        return "", optimized_prompt
    
    def _mcts_workflow(self, init_prompt: str):
        root_node = PromptNode(
                action_set=self.actions,
                action_seq=[],
                trajectory_prompts=[],
                prompt=init_prompt,
                evaluator=self.evaluator,
                depth=0,
                sample_pool=None
            )
        
        visualizer = MCTSVisualizer(root_node)
        visualizer.start()

        mcts_iters = self.config.mcts_iter_num_max
        expand_width = self.config.width_threshold

        mcts = MCTS(
            iter_num=mcts_iters,
            max_depth=self.config.max_depth_threshold,
            min_depth=self.config.min_depth_threshold,
            expand_width=expand_width, 
            exploration_weight=self.config.exploration_weight,

            expand_strategy=get_expand_strategy(self.config),
            rollout_strategy=get_rollout_strategy(self.config),
            choose_strategy=get_choose_strategy(self.config)
        )
        mcts.min_reward_threshold = root_node.reward_value
        mcts.increase_threshold(root_node.reward_value)
        logger.info("🏁 Start Search.")
        for iter_id in range(mcts_iters):
            mcts.do_iter(root_node, iter_id)

        best_node: PromptNode = mcts.choose(root_node)
        logger.info("🏁 Search completed. Selected best action sequence:")
        for i, action in enumerate(best_node.action_seq):
            logger.info(f"  Step {i+1}: {action.name}")
        best_prompt = best_node.current_prompt

        result_dict = mcts.serialize(root_node)

        os.makedirs("logs", exist_ok=True)
        with open(f"logs/{self.task.name}_promptagent_mcts_full_tree.json", "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        logger.info("✅ Full MCTS tree has been saved to logs/promptagent_mcts_full_tree.json")

        return best_prompt