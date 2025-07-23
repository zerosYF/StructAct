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
from multiprocessing.dummy import Pool
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

        Visualizer.start(title=self.task.name)
        
        for _ in range(self.config.rnn_iter_num):
            best_prompt = template.pre_sample(best_prompt)
            self._mcts_workflow(template, best_prompt)
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

        Visualizer.set_mcts(mcts, root_node)

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
    
    def batch_search(self):
        template = PromptTemplate(config=self.config, blocks=self.blocks, task=self.task)
        logger.info(f"🔍 Initial template constraints:\n{template.render()}")
        best_prompt = self.task.extract_origin_prompt()

        Visualizer.start(title=self.task.name)
        
        for _ in range(self.config.rnn_iter_num):
            samples = template.batch_sample_structs(self.config.struct_sample_count)
            top_k = template.select_topk_structures(samples, self.config.struct_sample_top_k)
            logger.info(f"📌 Top-{self.config.struct_sample_top_k} structures selected. Starting parallel MCTS...")
            top1 = top_k[0]
            others = top_k[1:]
            main_result = self._mcts_workflow_for_batch(
                [self.blocks, self.task, top1[1], best_prompt, self.evaluator, self.config], True
            )
            other_results = []
            if self.config.struct_sample_top_k > 1:
                args_list = [
                (self.blocks, self.task, flat_params, best_prompt, self.evaluator, self.config, False)
                    for (_, flat_params, _, _) in others
                ]
                with Pool(processes=self.config.struct_sample_top_k) as pool:
                    other_results = pool.map(self._mcts_workflow_for_batch, args_list)
            
            all_results = [main_result] + other_results
            for result in all_results:
                for _, flat_params, log_prob_sum, entropy in top_k:
                    if tuple(result["flat_params"]) == tuple(flat_params):
                        result["log_prob_sum"] = log_prob_sum
                        result["entropy"] = entropy
                        break

            best_result = max(all_results, key=lambda x: x["reward"])
            best_prompt = best_result["prompt"]

            template.last_sampled_params = best_result["flat_params"]
            template.last_log_prob_sum = best_result["log_prob_sum"]
            template.last_entropy = best_result["entropy"]
            template.update(self.evaluator, best_prompt)
            
        return template.render(), best_prompt

    def _mcts_workflow_for_batch(self, args, main_thread=False):
        blocks, task, flat_params, prompt, evaluator, config = args

        template = PromptTemplate(config=config, blocks=blocks, task=task)
        prompt = template.update_params(flat_params, prompt)

        root_node = PromptNode(
            action_set=define_full_actions(task),
            action_seq=[],
            structure_template=template,
            prompt=prompt,
            evaluator=evaluator,
            depth=0,
            max_depth=config.depth_threshold,
        )

        mcts = MCTS(
            select_strategy=get_select_strategy(config),
            expand_strategy=get_expand_strategy(config),
            rollout_strategy=get_rollout_strategy(evaluator, config),
            choose_strategy=get_choose_strategy(config)
        )
        if main_thread:
            Visualizer.set_mcts(mcts, root_node)

        for _ in range(config.mcts_iter_num):
            mcts.do_iter(
                root_node,
                width=config.width_threshold,
                expand_num=config.expand_num,
            )
        best_node = mcts.choose(root_node)
        return {
            "prompt": best_node.current_prompt,
            "reward": template.get_reward(evaluator, best_node.current_prompt),
            "flat_params": flat_params,
            "log_prob_sum": 0.0, 
            "entropy": 0.0
        }