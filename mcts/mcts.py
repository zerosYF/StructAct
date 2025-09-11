from typing import Any
from mcts.expand import ExpandStrategy
from mcts.choose import ChooseStrategy
from mcts.rollout import RolloutStrategy
from logger import logger
from mcts.node import Node
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

class MCTS:
    def __init__(self, 
                 iter_num:int,
                 expand_width:int, 
                 rollout_length:int,
                 exploration_weight:float,
                 max_depth:int,
                 min_depth:int,

                 expand_strategy: ExpandStrategy = None,
                 rollout_strategy: RolloutStrategy = None,
                 choose_strategy: ChooseStrategy = None, 
                ):
        self.iter_num = iter_num

        self.global_highest_reward:float = 0.0
        self.min_reward_threshold:float = 0.0

        self.exploration_weight = exploration_weight
        self.expand_width = expand_width
        self.rollout_length = rollout_length
        self.max_depth = max_depth
        self.min_depth = min_depth

        self.expand_strategy = expand_strategy
        self.rollout_strategy = rollout_strategy
        self.choose_strategy = choose_strategy
        self.lock = Lock()

    def _select(self, node: Node, exploration_weight=1.41) -> list[Node]:
        while not node.is_leaf() and not self.is_terminal_node(node):
            node = node.find_best_node(exploration_weight)
        return node

    def _expand(self, node: Node, expand_width=1) -> Node:
        return self.expand_strategy.expand(node, expand_width, self)

    def _rollout(self, node: Node, rollout_length=3):
        return self.rollout_strategy.rollout(node, rollout_length, self)

    def _backpropagate(self, expand_node:Node, avg_reward:float):
        with self.lock:
            current = expand_node
            while current is not None:
                current.update(avg_reward)
                current = current.parent


    def do_iter(self, root: Node, iter_id:int):
        logger.info(f"--------------Start Iteration:{iter_id}----------------")
        logger.info("Step 1: Performing Select")
        selected_node:Node = self._select(root, self.exploration_weight)
        logger.info(f"Selected leaf node type: {selected_node.type}")
        logger.info("Step 2: Performing Expand")
        children = self._expand(selected_node, self.expand_width)
        rollout_targets = children if children else [selected_node]

        results = []
        logger.info("Step 3: Performing Rollout")
        with ThreadPoolExecutor(max_workers=len(rollout_targets)) as executor:
            future_to_child = {
                executor.submit(self._rollout, child, self.rollout_length): child
                for child in rollout_targets
            }
            for future in as_completed(future_to_child):
                child = future_to_child[future]
                try:
                    avg_reward = future.result()
                    results.append((child, avg_reward))
                except Exception as e:
                    logger.error(f"⚠️ Error during rollout of {child}: {e}")

        logger.info("Step 4: Performing Backpropagate")
        for child, avg_reward in results:
            self._backpropagate(child, avg_reward)

        logger.info("---------------End Iteration------------------")

    def choose(self, root: Node) -> Node:
        return self.choose_strategy.choose(root)
    
    #########Rule################################################################################################################
    
    def is_terminal_node(self, node:Node):
        return node.is_terminal or node.depth >= self.max_depth or self.is_terminal_with_min_threshold(node)
    
    def increase_threshold(self, threshold):
        if threshold >= self.global_highest_reward:
            self.global_highest_reward = threshold
    
    def should_early_stop(self, node:Node):
        '''贪心早停'''
        logger.info(f"[Early Stop] early_stop:{node.type}")
        return node.reward_value > self.global_highest_reward and node.depth > self.min_depth
    
    def is_terminal_with_min_threshold(self, node: Node):
        if node.parent is None:
            min_threshold = self.min_reward_threshold
        else:
            min_threshold = (self.min_reward_threshold + node.parent.reward_value) / 2
        return node.reward_value < min_threshold and node.depth > self.min_depth

    ########Log####################################################################################################################
    def serialize(self, root:Node):
        best_node = self.choose(root)
        init_id = 0
        result_dict = {
            "config": {
                "mcts_iters": self.iter_num,
                "depth_threshold": self.max_depth,
                "width_threshold": self.expand_width,
            },
            "best_node": {
                "action_sequence": [a.name for a in best_node.action_seq],
                "prompt": best_node.current_prompt,
                "depth": best_node.depth,
                "Q": best_node.Q,
                "N": best_node.N,
            },
            "search_tree": root.serialize(init_id)
        }
        return result_dict
        