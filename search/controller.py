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
import concurrent.futures
import math
from logger import logger

class SearchController:
    def __init__(self, 
                 evaluator: PromptEvaluator, 
                 config: SearchConfig, 
                 task: TaskBase):
        self.actions: Set[OptimizeAction] = define_full_actions(task)
        self.evaluator: PromptEvaluator = evaluator
        self.config: SearchConfig = config
        self.task: TaskBase = task

    def search(self):
        template = PromptTemplate(config=self.config, blocks=get_all_blocks(), task=self.task)
        logger.info(f"🔍 Initial template constraints:\n{template.render()}")
        best_prompt = self.task.extract_origin_prompt()

        Visualizer.start(title=self.task.name)
        
        for epoch in range(self.config.rnn_iter_num):
            best_prompt = template.pre_sample(best_prompt)
            best_prompt = self._mcts_workflow(template, best_prompt, epoch)
            template.update(self.evaluator, best_prompt)
        return template.render(), best_prompt
    
    
    def _mcts_workflow(self, template: PromptTemplate, best_prompt: str, current_rnn_iter: int = 0, main_thread: bool = True):
        root_node = PromptNode(
                action_set=self.actions,
                action_seq=[],
                structure_template=template,
                prompt=best_prompt,
                evaluator=self.evaluator,
                depth=0,
                max_depth=self.config.depth_threshold,
            )

        mcts_iters = self.schedule_mcts_iter(current_rnn_iter, self.config.rnn_iter_num, self.config.mcts_iter_num_min, self.config.mcts_iter_num_max)
        rollout_len = self.schedule_rollout_length(current_rnn_iter, self.config.rnn_iter_num, self.config.rollout_length_min, self.config.rollout_length_max)
        expand_count = self.schedule_expand_num(current_rnn_iter, self.config.rnn_iter_num, self.config.expand_num_min, self.config.expand_num_max)
        print(f"MCTS Iterations: {mcts_iters}, Rollout Length: {rollout_len}, Expand Count: {expand_count}")

        mcts = MCTS(
            select_strategy=get_select_strategy(self.config),
            expand_strategy=get_expand_strategy(expand_count),
            rollout_strategy=get_rollout_strategy(self.evaluator, rollout_len, self.config),
            choose_strategy=get_choose_strategy(self.config)
        )
        if main_thread:
            Visualizer.set_mcts(mcts, root_node)

        for iter_id in range(mcts_iters):
            mcts.do_iter(root_node, width=self.config.width_threshold)
            if iter_id % 5 == 0 or iter_id == mcts_iters - 1:
                logger.info(f"  Total expanded nodes: {len(mcts.N)}")

        best_node: PromptNode = mcts.choose(root_node)
        logger.info("🏁 Search completed. Selected best action sequence:")
        for i, action in enumerate(best_node.action_seq):
            logger.info(f"  Step {i+1}: {action.name}")
        best_prompt = best_node.current_prompt
        return best_prompt
    
    def batch_search(self):
        template = PromptTemplate(config=self.config, blocks=get_all_blocks(), task=self.task)
        logger.info(f"🔍 Initial template constraints:\n{template.render()}")
        best_prompt = self.task.extract_origin_prompt()

        Visualizer.start(title=self.task.name)
        
        for epoch in range(self.config.rnn_iter_num):
            samples = template.batch_sample_structs(self.config.struct_sample_count)
            top_k = template.select_topk_structures(samples, self.config.struct_sample_top_k)
            logger.info(f"📌 Top-{self.config.struct_sample_top_k} structures selected. Starting parallel MCTS...")
            top1, others = top_k[0], top_k[1:]
            top1_init_prompt = template.update_params(top1[1], best_prompt)
            other_init_prompts = [template.update_params(params, best_prompt) for _, params, _, _ in others]

            top1_result_prompt = self._mcts_workflow(template, top1_init_prompt, epoch)
            other_result_prompts = []
            if self.config.struct_sample_top_k > 1:
                args_list = [
                (template, init_prompt, epoch, False)
                    for init_prompt in other_init_prompts
                ]
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.struct_sample_top_k) as executor:
                    other_result_prompts = list(executor.map(self._mcts_workflow, args_list))
            
            all_results = [top1_result_prompt] + other_result_prompts
            best_reward = float("-inf")
            # map keep index
            for i, candidate_best_prompt in enumerate(all_results):
                reward = template.get_reward(self.evaluator, candidate_best_prompt)
                if reward > best_reward:
                    best_reward = reward
                    best_prompt = candidate_best_prompt
                    template.last_sampled_params = top_k[i][1]
                    template.last_log_prob_sum = top_k[i][2]
                    template.last_entropy = top_k[i][3]
            template.update(self.evaluator, best_prompt)
            
        return template.render(), best_prompt
    
    def nonlinear_schedule(
        self,
        rnn_iter: int,
        total_iter: int,
        min_val: int,
        max_val: int,
        steepness: float = 10.0,
        pivot: float = 0.9
    ) -> int:
        """
        Nonlinear scheduling using a sigmoid curve.

        Args:
            rnn_iter (int): Current RNN iteration index.
            total_iter (int): Total RNN iterations.
            min_val (int): Minimum scheduled value.
            max_val (int): Maximum scheduled value.
            steepness (float): Controls curve steepness.
            pivot (float): Fraction [0,1] indicating where rapid increase starts.

        Returns:
            int: Scheduled integer value.
        """
        progress = rnn_iter / total_iter
        sigmoid = 1 / (1 + math.exp(-steepness * (progress - pivot)))
        value = min_val + (max_val - min_val) * sigmoid
        return max(min_val, min(max_val, int(round(value))))

    def schedule_mcts_iter(self, rnn_iter: int, total_iter: int, min_val=1, max_val=10) -> int:
        return self.nonlinear_schedule(rnn_iter, total_iter, min_val=min_val, max_val=max_val, steepness=12.0, pivot=0.993)

    def schedule_rollout_length(self, rnn_iter: int, total_iter: int, min_val=1, max_val=5) -> int:
        return self.nonlinear_schedule(rnn_iter, total_iter, min_val=min_val, max_val=max_val, steepness=12.0, pivot=0.993)

    def schedule_expand_num(self, rnn_iter: int, total_iter: int, min_val=0, max_val=3) -> int:
        return self.nonlinear_schedule(rnn_iter, total_iter, min_val=min_val, max_val=max_val, steepness=12.0, pivot=0.993)