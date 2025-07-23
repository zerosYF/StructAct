from task.base_task import TaskBase
from program.base_action import OptimizeAction
from search.evaluator import PromptEvaluator
from mcts.mcts import MCTS
from search.prompt_node import PromptNode
from search.config import SearchConfig
from typing import List, Set
from visualizer import Visualizer
from program.base_block import PromptBlock
from program.prompt_template import PromptTemplate
from program.good_blocks import get_all_blocks
from mcts.select import get_select_strategy
from mcts.expand import get_expand_strategy
from mcts.rollout import get_rollout_strategy
from mcts.choose import get_choose_strategy
from program.strategy_actions import define_full_actions
from logger import logger

class SearchController:
    def __init__(self, 
                 evaluator: PromptEvaluator, 
                 config: SearchConfig, 
                 task: TaskBase):
        self.actions: Set[OptimizeAction] = define_full_actions(task)
        self.blocks: List[PromptBlock] = get_all_blocks()
        self.evaluator: PromptEvaluator = evaluator
        self.config: SearchConfig = config
        self.task: TaskBase = task

    def search(self):
        template = PromptTemplate(config=self.config, blocks=self.blocks, task=self.task)
        logger.info(f"🔍 Initial template constraints:\n{template.render()}")
        best_prompt = self.task.extract_origin_prompt()
        
        for _ in range(self.config.rnn_iter_num):
            best_prompt, need_mcts = template.pre_sample(best_prompt)
            if need_mcts:
                best_prompt = self._mcts_workflow(template, best_prompt)
            template.update(self.evaluator, best_prompt)
        return template.render(), best_prompt
    
    
    def _mcts_workflow(self, template: PromptTemplate, best_prompt: str):
        root_node = PromptNode(
                action_set=self.actions,
                action_seq=[],
                structure_template=template,
                prompt=best_prompt,
                evaluator=self.evaluator,
                depth=0,
                max_depth=self.config.depth_threshold,
            )
            
        logger.info(f"🔍 Starting MCTS search with {self.config.mcts_iter_num} iterations, max path depth = {self.config.depth_threshold}")

        mcts = MCTS(
            select_strategy=get_select_strategy(self.config),
            expand_strategy=get_expand_strategy(self.config),
            rollout_strategy=get_rollout_strategy(self.evaluator, self.config),
            choose_strategy=get_choose_strategy(self.config)
        )

        Visualizer.set_mcts(mcts, root_node, title=self.task.name)
        Visualizer.start()

        for iter_id in range(self.config.mcts_iter_num):
            mcts.do_iter(
                root_node,
                width=self.config.width_threshold,
                expand_num=self.config.expand_num,
            )
            if iter_id % 5 == 0 or iter_id == self.config.mcts_iter_num - 1:
                logger.info(f"  Total expanded nodes: {len(mcts.N)}")

        best_node: PromptNode = mcts.choose(root_node)
        logger.info("🏁 Search completed. Selected best action sequence:")
        for i, action in enumerate(best_node.action_seq):
            logger.info(f"  Step {i+1}: {action.name}")
        best_prompt = best_node.current_prompt
        return best_prompt