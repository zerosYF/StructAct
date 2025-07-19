from typing import List
from logger import logger
from rnn.controller import TemplateController
from search.evaluator import PromptEvaluator
from search.config import SearchConfig
from program.base_block import PromptBlock
from program.base_action import StructureSyncAction
from task.base_task import TaskBase

class PromptTemplate:
    def __init__(self, config:SearchConfig, blocks: List[PromptBlock], task:TaskBase = None):
        self.blocks = blocks
        self.controller = TemplateController(search_space=self._get_serch_space(),
                                             hidden_dim=config.rnn_hidden_dim,
                                             lr=config.rnn_lr)
        self.task = task
        self.sync_action = StructureSyncAction(task, self.task.extract_origin_prompt())
    
    def _get_serch_space(self) -> List[int]:
        """
        获取所有 block 的搜索空间
        """
        search_space = []
        for block in self.blocks:
            search_space.extend(block.get_search_space())
        return search_space

    def render(self) -> str:
        return "\n".join([f"{block.render()}" for block in self.blocks])

    def describe(self) -> str:
        return "\n".join([block.describe() for block in self.blocks])

    def update_by_controller(self, evaluator: PromptEvaluator, current_prompt: str):
        # Step 1: 控制器生成结构参数
        flat_params, log_prob_sum, entropy = self.controller.train_step()

        # Step 2: 设置参数到各个 block 中
        idx = 0
        for block in self.blocks:
            num = block.get_num_slots()
            block.set_hyperparams(flat_params[idx:idx + num])
            idx += num

        # ✅ Step 3: 根据新的 block 参数，重新生成 prompt
        new_prompt = self._sync_semantics(current_prompt)

        # Step 4: 用新 prompt 打分
        val_samples = self.task.get_val_rl()
        total_score = sum(evaluator.batch_reward(new_prompt, val_samples))
        avg_score = total_score / len(val_samples)
        logger.info(f"🎯 [PromptTemplate] 使用当前结构超参数得到的新prompt得分 = {avg_score:.4f}")

        # Step 5: reinforce 更新 controller
        self.controller.reinforce(log_prob_sum, avg_score, entropy)

        return new_prompt
    
    def _sync_semantics(self, current_prompt) -> str:
        """
        使用专门的 SyncAction，根据结构模板和样本，生成语义完整的 prompt
        """
        template_description = self.describe()
        return self.sync_action.do(
            current_prompt=current_prompt,
            template_description=template_description,
        )