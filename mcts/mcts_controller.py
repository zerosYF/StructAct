from task.base_task import TaskBase
from search.action import OptimizeAction
from search.evaluator import PromptEvaluator
from mcts.mcts import MCTS
from Experiment.mcts.prompt_node import PromptNode
from search.config import SearchConfig
from typing import List, Set
from visualizer import Visualizer
from rnn.rnn_controller import RNNController
from rnn.template import PromptTemplate
from rnn.blocks import get_all_blocks
from logger import logger

class MCTSController:
    def __init__(self, 
                 evaluator:PromptEvaluator, 
                 config:SearchConfig, 
                 task:TaskBase, 
                 actions:Set[OptimizeAction]
                 ):
        self.actions:Set[OptimizeAction] = actions
        self.evaluator:PromptEvaluator = evaluator
        self.config:SearchConfig= config
        self.task:TaskBase = task
    
    def search(self):
        blocks = get_all_blocks()
        rnn_controller = RNNController(
                blocks=blocks, 
                hidden_dim=self.config.rnn_hidden_dim, lr=self.config.rnn_lr)
        template = PromptTemplate(controller=rnn_controller, blocks=get_all_blocks())

        root_node = PromptNode(action_set=self.actions, 
                               action_seq=[], 
                               structure_template=template,
                               prompt=self.task.extract_origin_prompt(), 
                               evaluator=self.evaluator, 
                               depth=0, 
                               max_depth=self.config.depth_threshold,
                               )
        logger.info(f"🔍 开始 MCTS 搜索，共 {self.config.iter_num} 次迭代，每条路径最大深度为 {self.config.depth_threshold}")


        choose_strategy = self.config.get_choose_strategy()
        rollout_strategy = self.config.get_rollout_strategy(self.evaluator)
        uct_strategy = self.config.get_uct_strategy()

        mcts = MCTS(choose_strategy=choose_strategy, rollout_strategy=rollout_strategy, uct_strategy=uct_strategy)

        Visualizer.set_mcts(mcts, root_node)
        Visualizer.start()

        for iter_id in range(self.config.iter_num):
            mcts.do_iter(root_node, self.config.expand_num, self.config.rollout_parallel)

            if iter_id % 5 == 0 or iter_id == self.config.iter_num - 1:
                logger.info(f"  当前累计节点数: {len(mcts.N)}")
        
        #visualizer.stop()

        best_node:PromptNode = mcts.choose(root_node)
        logger.info("🏁 搜索结束，选择得分最高的动作序列:")
        for i, action in enumerate(best_node.action_seq):
            logger.info(f"  Step {i+1}: {action.name}")
        
        return best_node.structure_template.render(), best_node.action_seq, best_node.current_prompt