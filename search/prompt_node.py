from program.base_action import OptimizeAction
from search.evaluator import PromptEvaluator
from mcts.node import Node, Step
from program.sample_pools import DynamicSamplePool
from typing import List, Set
from logger import logger

class PromptNode(Node):
    def __init__(self, 
                 action_set: Set[OptimizeAction],
                 action_seq: List[OptimizeAction], 
                 trajectory_prompts: List[str],
                 prompt: str, 
                 evaluator: PromptEvaluator, 
                 depth: int, 
                 max_depth: int,
                 sample_pool: DynamicSamplePool = None):
        
        self.type = prompt
        self.action_set: Set[OptimizeAction] = action_set
        self.evaluator: PromptEvaluator = evaluator
        self.trajectory_prompts: List[str] = trajectory_prompts
        self.current_prompt: str = prompt
        logger.info(f"📜 Get new node at depth {depth} using action {action_seq[-1].name if len(action_seq) > 0 else None}")
        
        self.children: List[PromptNode] = None
        self.action_seq: List[OptimizeAction] = action_seq

        self.depth = depth
        self.max_depth = max_depth
        self.reward_value: float = self.reward()
        self.pool = sample_pool

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def get_untried_actions(self):
        # Used only once when initializing a new MCTS node.
        # Controller will be responsible for popping later.
        return list(self.action_set)

    def get_possible_actions(self):
        # In each rollout, actions can be reselected.
        return list(self.action_set)

    def take_action(self, action: OptimizeAction, step_type:Step):
        # Then apply the strategy-level semantic transformation.
        new_prompt = action.do(
            current_prompt=self.current_prompt, 
            trajectory_prompts=self.trajectory_prompts, 
            sample_pool=self.pool
            )
        logger.info(f"📊 Current Prompt:\n{new_prompt}")
        return PromptNode(
            action_set=self.action_set,
            action_seq=self.action_seq + [action],
            trajectory_prompts=self.trajectory_prompts + [self.current_prompt],
            prompt=new_prompt,
            evaluator=self.evaluator,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            sample_pool=self.pool
        )

    def is_terminal(self):
        return self.depth == self.max_depth

    def reward(self):
        val_samples = self.evaluator.task.get_eval()
        score = self.evaluator.batch_reward(self.current_prompt, val_samples)
        logger.info(f"🎯 [Reward] Prompt evaluation score = {score:.4f}, Action sequence = {[a.name for a in self.action_seq]}")
        return score

    def clone_node(self):
        return PromptNode(
            action_set=self.action_set,  # shared reference
            action_seq=list(self.action_seq),
            trajectory_prompts=list(self.trajectory_prompts),
            prompt=self.current_prompt,
            evaluator=self.evaluator,
            depth=self.depth,
            max_depth=self.max_depth,
            sample_pool=self.pool
        )
    
    def q_value(self, last_q, rollout_reward):
        return last_q + rollout_reward