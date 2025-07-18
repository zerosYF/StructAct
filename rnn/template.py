from typing import List
from logger import logger
from rnn.rnn_controller import RNNController
from search.evaluator import PromptEvaluator
from rnn.blocks import PromptBlock
from search.action import StructureSyncAction
from task.base_task import TaskBase
from rnn.rnn_controller import RNNController

class PromptTemplate:
    def __init__(self, controller:RNNController, blocks: List[PromptBlock], task:TaskBase = None):
        self.controller = controller
        self.blocks = blocks
        self.task = task
        self.sync_action = StructureSyncAction(task)

    def render(self) -> str:
        return "\n".join([block.render() for block in self.blocks])

    def describe(self) -> str:
        return "\n".join([block.describe() for block in self.blocks])

    def update_by_rnn(self, evaluator: PromptEvaluator, current_prompt: str):
        # Step 1: 从 controller 中采样结构参数
        flat_params, log_prob_sum, entropy = self.controller.train_step()

        # Step 2: 设置参数到各个 block 中
        idx = 0
        for block in self.blocks:
            num = block.get_num_slots()
            block.set_hyperparams(flat_params[idx:idx + num])
            idx += num

        # Step 3: 渲染完整 prompt（作为结构提示）
        structure_prompt = self.render()

        # Step 4: 用当前 prompt 在 evaluator 上打分（可自定义评分方式）
        val_samples = evaluator.task.get_val()
        total_score = sum(evaluator.batch_reward(current_prompt, val_samples))
        avg_score = total_score / len(val_samples)

        # Step 5: 执行一次 reinforce 更新 controller
        self.controller.reinforce(log_prob_sum, avg_score, entropy)

        logger.info(f"🎯 [PromptTemplate] 使用结构超参数更新，得分 = {avg_score:.4f}")
    
    def sync_semantics(self, current_prompt) -> str:
        """
        使用专门的 SyncAction，根据结构模板和样本，生成语义完整的 prompt
        """
        template_description = self.describe()
        return self.sync_action.do(
            current_prompt=current_prompt,
            template_structure=template_description,
        )