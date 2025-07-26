from program.base_action import OptimizeAction
from search.evaluator import PromptEvaluator
from mcts.node import Node, Step
from program.prompt_template import PromptTemplate
from typing import List, Set
from logger import logger

class PromptNode(Node):
    def __init__(self, 
                 action_set: Set[OptimizeAction],
                 action_seq: List[OptimizeAction], 
                 structure_template: PromptTemplate,
                 prompt: str, 
                 evaluator: PromptEvaluator, 
                 depth: int, 
                 max_depth: int):
        
        self.type = prompt
        self.action_set: Set[OptimizeAction] = action_set
        self.structure_template: PromptTemplate = structure_template
        self.evaluator: PromptEvaluator = evaluator
        self.current_prompt: str = prompt
        logger.info(f"📜 Get new node at depth {depth} using action {action_seq[-1].name if len(action_seq) > 0 else None}")
        
        self.children: List[PromptNode] = None
        self.action_seq: List[OptimizeAction] = action_seq

        self.depth = depth
        self.max_depth = max_depth

    def __hash__(self):
        return hash(tuple(self.action_seq))

    def __eq__(self, other):
        return isinstance(other, PromptNode) and self.action_seq == other.action_seq

    def get_untried_actions(self):
        # Used only once when initializing a new MCTS node.
        # Controller will be responsible for popping later.
        return list(self.action_set)

    def get_possible_actions(self):
        # In each rollout, actions can be reselected.
        return list(self.action_set)

    def take_action(self, action: OptimizeAction, step_type:Step):
        # Then apply the strategy-level semantic transformation.
        new_prompt = action.do(self.current_prompt, self.structure_template.render())
        logger.info(f"📊 Current Prompt:\n{new_prompt}")
        return PromptNode(
            action_set=self.action_set,
            action_seq=self.action_seq + [action],
            structure_template=self.structure_template,
            prompt=new_prompt,
            evaluator=self.evaluator,
            depth=self.depth + 1,
            max_depth=self.max_depth,
        )

    def is_terminal(self):
        return self.depth == self.max_depth

    def reward(self):
        val_samples = self.evaluator.task.get_eval()
        total_score = sum(self.evaluator.batch_reward(self.current_prompt, val_samples))
        score = total_score / len(val_samples)
        logger.info(f"🎯 [RolloutFinished] Prompt evaluation score = {score:.4f}, Action sequence = {[a.name for a in self.action_seq]}")
        return score

    def clone_node(self):
        return PromptNode(
            action_set=self.action_set,  # shared reference
            action_seq=list(self.action_seq),
            structure_template=self.structure_template,
            prompt=self.current_prompt,
            evaluator=self.evaluator,
            depth=self.depth,
            max_depth=self.max_depth,
        )
    
    def q_value(self, last_q, rollout_reward):
        return last_q + (rollout_reward ** 2)