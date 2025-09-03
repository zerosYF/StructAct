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
from logger import logger

class MCTSearchController(SearchController):
    def __init__(self, 
                 evaluator: PromptEvaluator, 
                 config: SearchConfig, 
                 task: TaskBase):
        super().__init__(evaluator, config, task)
        self.actions: Set[OptimizeAction] = define_full_actions(task)

    def search(self):
        init_prompt = self.task.origin_prompt

        Visualizer.start(title=self.task.name)
        optimized_prompt = self._mcts_workflow(init_prompt)
        return optimized_prompt
    
    def _mcts_workflow(self, best_prompt: str):
        root_node = PromptNode(
                action_set=self.actions,
                action_seq=[],
                prompt=best_prompt,
                evaluator=self.evaluator,
                depth=0,
                max_depth=self.config.depth_threshold,
            )

        mcts_iters = self.config.mcts_iter_num_max
        rollout_len = self.config.rollout_length_max
        expand_count = self.config.expand_num_max
        print(f"MCTS Iterations: {mcts_iters}, Rollout Length: {rollout_len}, Expand Count: {expand_count}")

        mcts = MCTS(
            select_strategy=get_select_strategy(self.config),
            expand_strategy=get_expand_strategy(self.config),
            rollout_strategy=get_rollout_strategy(self.evaluator, self.config),
            choose_strategy=get_choose_strategy(self.config)
        )

        Visualizer.set_mcts(mcts, root_node)

        for iter_id in range(mcts_iters):
            mcts.do_iter(root_node, 
                              expand_width=expand_count, 
                              rollout_length=rollout_len, 
                              select_width=self.config.width_threshold)
            if iter_id % 5 == 0 or iter_id == mcts_iters - 1:
                logger.info(f"  Total expanded nodes: {len(mcts.N)}")

        best_node: PromptNode = mcts.choose(root_node)
        logger.info("🏁 Search completed. Selected best action sequence:")
        for i, action in enumerate(best_node.action_seq):
            logger.info(f"  Step {i+1}: {action.name}")
        best_prompt = best_node.current_prompt
        return best_prompt