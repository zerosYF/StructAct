from program.base_action import OptimizeAction
from search.evaluator import PromptEvaluator
from mcts.node import Node, Step
from program.sample_pools import DynamicSamplePool
from typing import List, Set
from logger import logger
import numpy as np

class PromptNode(Node):
    def __init__(self, 
                 action_set: Set[OptimizeAction],
                 action_seq: List[OptimizeAction], 
                 trajectory_prompts: List[str],
                 prompt: str, 
                 evaluator: PromptEvaluator, 
                 depth: int, 

                 sample_pool: DynamicSamplePool = None,

                 Q:float=0.0, N:int=0, 
                 uct_value:float=0.0, 
                 parent=None
                 ):
        super().__init__(depth=depth, Q=Q, N=N, uct_value=uct_value, parent=parent)
        self.type = prompt
        self.action_set: Set[OptimizeAction] = action_set
        self.evaluator: PromptEvaluator = evaluator
        self.trajectory_prompts: List[str] = trajectory_prompts
        self.current_prompt: str = prompt
        logger.info(f"📜 Get new node at depth {depth} using action {action_seq[-1].name if len(action_seq) > 0 else None}")
        
        self.action_seq: List[OptimizeAction] = action_seq
        self.pool = sample_pool
        self.reward_value: float = self.reward()
        self.Q = self.reward_value
    
    def _weighted_random_choice(self, temperature: float = 1.0):
        """Softmax weighted random selection of a single action based on failure_counter and usage_count"""
        actions = list(self.action_set)

        failure_counts = np.array([a.sample_failure_counter for a in actions])
        usage_counts = np.array([a.usage_count for a in actions])

        logits = - (failure_counts + usage_counts) / temperature 
        logits /= temperature  

        probs = np.exp(logits)
        probs /= probs.sum() 

        selected_index = np.random.choice(len(actions), p=probs)
        
        return actions[selected_index]

    def take_action(self, step_type:Step):
        # Then apply the strategy-level semantic transformation.
        action:OptimizeAction = self._weighted_random_choice()
        new_prompt = action.do(
            current_prompt=self.current_prompt, 
            trajectory_prompts=self.trajectory_prompts, 
            sample_pool=self.pool
            )
        logger.info(f"📊 Current Step Type:{step_type}")
        logger.info(f"📊 Current Prompt:\n{new_prompt}")
        new_child = PromptNode(
            action_set=self.action_set,
            action_seq=self.action_seq + [action],
            trajectory_prompts=self.trajectory_prompts + [self.current_prompt],
            prompt=new_prompt,
            evaluator=self.evaluator,
            sample_pool=self.pool,

            depth=self.depth + 1,
            parent=self,
        )
        return new_child

    def reward(self):
        val_samples = self.evaluator.task.get_eval()
        score = self.evaluator.batch_reward(self.current_prompt, val_samples)
        logger.info(f"🎯 [Reward] Prompt evaluation score = {score:.4f}, Action sequence = {[a.name for a in self.action_seq]}")
        return score + self.pool.compute_cpool() if self.pool else score
    
    def q_value(self, last_q, rollout_reward):
        return last_q + rollout_reward
    
    def serialize(self, node_id:int):
        node_dict = {
            "id": node_id,
            "depth": self.depth,
            "action_sequence": [a.name for a in self.action_seq],
            "prompt": self.current_prompt,
            "Q": self.Q,
            "N": self.N, 
            "uct_value": self.uct_value,
            "reward": self.reward_value,
            "children": []
        }

        for child in self.children:
            node_id += 1
            child_serialized = child.serialize(node_id)
            if child_serialized:
                node_dict["children"].append(child_serialized)

        return node_dict