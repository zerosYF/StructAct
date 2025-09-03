from typing import List, Tuple, Any
from logger import logger
from rnn.controller import TemplateController
from search.evaluator import PromptEvaluator
from search.config import SearchConfig
from structs.base_block import PromptBlock
from program.base_action import StructureSyncAction
from task.base_task import TaskBase
import time

class History:
    def __init__(self, prompt:str, reward:float):
        self.prompt:str = prompt
        self.reward:float = reward

class PromptTemplate:
    def __init__(self, config: SearchConfig, blocks: List[PromptBlock], task: TaskBase = None):
        self.blocks = blocks
        self.controller = TemplateController(
            search_space=self._get_search_space(),
            hidden_dim=config.rnn_hidden_dim,
            lr=config.rnn_lr,
            reward_scale=config.rnn_rl_reward_scale,
        )
        self.task = task
        self.sync_action = StructureSyncAction(task, self.task.origin_prompt)
        self.struct_reward_cache:dict[tuple, History] = {}
        self.last_sampled_params = None
        self.evaluator = None  # Will be set when update is called

    def _get_search_space(self) -> List[int]:
        """
        Gather the full search space across all blocks.
        """
        search_space = []
        for block in self.blocks:
            search_space.extend(block.get_search_space())
        return search_space

    def describe(self) -> str:
        """
        Render each block as a JSON-like string and join them.
        """
        return "\n".join([f"{block.describe()}" for block in self.blocks])

    def render(self) -> str:
        """
        Generate a structured natural language prompt template with explicit markers.
        """
        template_header = """
            "This is current prompt template description:\n"
            "Each block is indicated by tags like <BLOCK_NAME>.\n"
            "The template also contain control tags like <TEMPLATE> and </TEMPLATE>, which denote boundaries of the full prompt structure.\n"
            "You can optimize the natural language content between and around these block and control tags, with the following requirements:\n"
            "1. Use the tags (e.g., <BLOCK_NAME>, <TEMPLATE>) **only as structural guidance during optimization**, but do not include them in output prompt;\n"
            "2. Do not alter the structure, order, or semantics implied by the original tags;\n"
            "3. The final output should be a **fully naturalized prompt**, with **constraint text about block, all tags and placeholders removed**;\n"
            "4. Ensure the resulting prompt is coherent, fluent, faithful to each block’s intent, and effective for the intended task.\n"
            "5. Make full use of information beyond the template description.\n"
            "This is current template:"
        """
        block_contents = "\n".join([block.render() for block in self.blocks])
        return f"{template_header}\n<TEMPLATE>\n{block_contents}\n</TEMPLATE>"
    
    def get_struct_history(self, params: List[float]) -> History:
        if tuple(params) in self.struct_reward_cache:
            return self.struct_reward_cache[tuple(params)]
        return None  
    
    def pre_sample(self, current_prompt: str):
        start_time = time.time()
        flat_params = self.controller.train_step()
        end_time = time.time()
        logger.info(f"⏱️ [PromptTemplate] RNN sample(infer) time: {end_time - start_time:.2f} seconds")

        self.last_sampled_params = flat_params
        updated_prompt, history_reward = self.update_params(flat_params, current_prompt)
        
        # If no cached reward, calculate it now
        if history_reward is None:
            if self.evaluator is not None:
                val_samples = self.task.get_train_rnn()
                history_reward = self.evaluator.batch_reward(updated_prompt, val_samples)
                # Cache the result
                self.struct_reward_cache[tuple(self.last_sampled_params)] = History(updated_prompt, history_reward)
                logger.info(f"🎯 [PromptTemplate] Calculated reward for new structure = {history_reward:.4f}")
            else:
                # Return 0 as initial reward if evaluator not set yet
                history_reward = 0.0
                logger.info(f"⚠️ [PromptTemplate] Evaluator not set, using default reward = {history_reward:.4f}")
            
        return updated_prompt, history_reward
    
    def update_params(self, flat_params: List[float], current_prompt: str) -> Tuple[str, float]:
        """
        Update the hyperparameters of each block based on the flat parameter vector.
        """
        idx = 0
        for block in self.blocks:
            num = block.get_num_slots()
            block.set_hyperparams(flat_params[idx:idx + num])
            idx += num
        history = self.get_struct_history(self.last_sampled_params)
        if history is None:
            start_time = time.time()
            current_prompt = self._sync_semantics(current_prompt)
            end_time = time.time()
            logger.info(f"⏱️ [PromptTemplate] LLM calling time: {end_time - start_time:.2f} seconds")
            # Return None or a special value to indicate reward needs to be calculated
            return current_prompt, None
        else:
            current_prompt = history.prompt
            return current_prompt, history.reward
    
    def get_reward(self, evaluator:PromptEvaluator,  current_prompt:str) -> float:
        val_samples = self.task.get_train_rnn()
        avg_score = evaluator.batch_reward(current_prompt, val_samples)
        if self.last_sampled_params is not None:
            self.struct_reward_cache[tuple(self.last_sampled_params)] = History(current_prompt, avg_score)

        logger.info(f"🎯 [PromptTemplate] New prompt score with current structure = {avg_score:.4f}")
        return avg_score

    def update(self, evaluator: PromptEvaluator, current_prompt: str, history_reward:float=None, skip_reinforce:bool=False):
        """
        Run one RNN optimization step:
        - Sample new structure parameters
        - Sync content to fit structure
        - Evaluate with reward
        - Compute slot-level attributions
        - Reinforce update
        """
        # Store evaluator for future use
        if self.evaluator is None:
            self.evaluator = evaluator
            
        # Skip reinforce if batch update was already done
        if skip_reinforce:
            return
            
        # current_prompt是一个列表
        # avg_score是多个（3个）prompt的平均分
        if history_reward is None or history_reward == -1e9:
            start_time = time.time()
            avg_score = self.get_reward(evaluator, current_prompt)
            end_time = time.time()
            logger.info(f"⏱️ [PromptTemplate] Reward evaluation time: {end_time - start_time:.2f} seconds")
        else:
            avg_score = history_reward

        # Ensure forward pass is done here to calculate log_prob_sum and entropy
        # Perform reinforcement learning update (the actual reinforcement step)
        start_time = time.time()
        self.controller.reinforce(avg_score)
        end_time = time.time()
        logger.info(f"⏱️ [PromptTemplate] RNN reinforce time: {end_time - start_time:.2f} seconds")

    def _sync_semantics(self, current_prompt: str) -> str:
        """
        Use the StructureSyncAction to regenerate a semantically complete prompt
        based on the current structural description and training examples.
        """
        return self.sync_action.do(
            current_prompt=current_prompt,
            template_description=self.render(),
        )
