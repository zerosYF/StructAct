from abc import ABC
from typing import List, Set, Tuple
import heapq
import itertools
import logging
from search.search import SearchController
from search.prompt_node import PromptNode
from program.sample_pools import DynamicSamplePool
from program.strategy_actions import OptimizeAction, define_full_actions

logger = logging.getLogger(__name__)

class BeamSearchController(SearchController):
    def __init__(self, evaluator, config, task):
        super().__init__(evaluator, config, task)
        self.beam_width = getattr(config, "beam_width", 5)
        self.max_depth = getattr(config, "max_depth", 5)
        self.pool: DynamicSamplePool = DynamicSamplePool(max_size=1000, low=0.5, high=0.9)
        self.actions: Set[OptimizeAction] = define_full_actions(task)
    
    def search(self):
        init_prompt = self.task.origin_prompt
        self.pool.initialize(self.task.get_train_mcts(), self.evaluator, init_prompt)

        # 初始化根节点
        root = PromptNode(
            action_set=self.actions,
            action_seq=[],
            trajectory_prompts=[],
            prompt=init_prompt,
            evaluator=self.evaluator,
            depth=0,
            sample_pool=self.pool
        )

        # 使用计数器保证 heap 元素唯一，避免 PromptNode 比较报错
        counter = itertools.count()
        beam = [(-root.reward_value, next(counter), root)]
        best_node = root
        best_score = root.reward_value

        for depth in range(self.max_depth):
            new_beam = []
            for _, _, node in beam:
                for _ in range(self.config.width_threshold):  
                    next_node = node.take_action(step_type=None)
                    if next_node is None:
                        continue  # 如果 take_action 可能失败，要跳过

                    score = next_node.reward_value
                    if score > best_score:
                        best_score = score
                        best_node = next_node

                    heapq.heappush(new_beam, (-score, next(counter), next_node))

            # 保留 top-k 节点
            beam = heapq.nsmallest(self.beam_width, new_beam)
            if not beam:
                break

        logger.info(f"✅ Beam Search Finished. Best score = {best_score:.4f}")
        return "", best_node.current_prompt