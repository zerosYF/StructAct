from typing import List, Tuple
from logger import logger
from rnn.controller import TemplateController
from search.evaluator import PromptEvaluator
from search.config import SearchConfig
from program.base_block import PromptBlock
from program.base_action import StructureSyncAction
from task.base_task import TaskBase

class PromptTemplate:
    def __init__(self, config: SearchConfig, blocks: List[PromptBlock], task: TaskBase = None):
        self.blocks = blocks
        self.controller = TemplateController(
            search_space=self._get_search_space(),
            hidden_dim=config.rnn_hidden_dim,
            lr=config.rnn_lr,
            reward_scale=config.rnn_rl_reward_scale,
            baseline=config.rnn_rl_baseline,
            baseline_alpha=config.rnn_rl_baseline_alpha,
            min_entropy_weight=config.rnn_rl_min_entropy_weight,
            max_entropy_weight=config.rnn_rl_max_entropy_weight,
            entropy_decay_rate=config.rnn_rl_entropy_decay_rate,
            attribution_interval=config.rnn_attribution_interval,
            aux_loss_coef=config.rnn_aux_loss_coef
        )
        self.task = task
        self.sync_action = StructureSyncAction(task, self.task.origin_prompt)
        self.struct_reward_cache = {}
        self.last_sampled_params = None
        self.last_log_prob_sum = 0.0
        self.last_entropy = 0.0

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
        instruction_header = (
            "This is current prompt template description:\n"
            "Each block is indicated by tags like <BLOCK_NAME> and parameterized with fields such as <KEY=VALUE>.\n"
            "The template also contain control tags like <TEMPLATE> and </TEMPLATE>, which denote boundaries of the full prompt structure.\n"
            "You can optimize the natural language content by using other context information.\n"
            "You can optimize the natural language content between and around these structure and control tags, with the following requirements:\n"
            "- Use the tags (e.g., <BLOCK_NAME>, <KEY=VALUE>, <TEMPLATE>) **only as structural guidance during optimization**, but do not include them in output prompt;\n"
            "- Do not alter the structure, order, or semantics implied by the original tags;\n"
            "- The final output should be a **fully naturalized prompt**, with **constraint text about block, all tags and placeholders removed**;\n"
            "- Ensure the resulting prompt is coherent, fluent, faithful to each block’s intent, and effective for the intended task."
        )

        block_contents = "\n".join([block.render() for block in self.blocks])

        return f"{instruction_header}\n<TEMPLATE>\n{block_contents}\n</TEMPLATE>"
    
    def batch_sample_structs(self, sample_k:int, oversample_factor: int = 2) -> list:
        seen = set()
        results = []

        max_trials = sample_k * oversample_factor
        trials = 0

        while len(results) < sample_k and trials < max_trials:
            flat_params, log_prob_sum, entropy = self.controller.train_step()
            key = tuple(flat_params)
            if key not in seen:
                seen.add(key)
                results.append((flat_params, log_prob_sum, entropy))
            trials += 1

        if len(results) < sample_k:
            logger.warning(f"[PromptTemplate] Only {len(results)} unique structures found out of {sample_k} requested.")

        return results
    
    def get_struct_reward(self, params: List[float]) -> float:
        if tuple(params) in self.struct_reward_cache:
            return self.struct_reward_cache[tuple(params)]
        return -1e9  
    
    def select_topk_structures(self, samples: List[Tuple[List[int], float, float]], k: int):
        scored = []
        for flat_params, log_prob_sum, entropy in samples:
            score = self.get_struct_reward(flat_params)
            scored.append((score, flat_params, log_prob_sum, entropy))
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:min(k, len(scored))]
    
    def pre_sample(self, current_prompt: str):
        flat_params, log_prob_sum, entropy = self.controller.train_step()

        self.last_sampled_params = flat_params
        updated_prompt = self.update_params(flat_params, current_prompt)
        self.last_log_prob_sum = log_prob_sum
        self.last_entropy = entropy
        return updated_prompt
    
    def update_params(self, flat_params: List[float], current_prompt: str) -> str:
        """
        Update the hyperparameters of each block based on the flat parameter vector.
        """
        idx = 0
        for block in self.blocks:
            num = block.get_num_slots()
            block.set_hyperparams(flat_params[idx:idx + num])
            idx += num
        current_prompt = self._sync_semantics(current_prompt)
        return current_prompt
    
    def get_reward(self, evaluator:PromptEvaluator,  current_prompt:str) -> float:
        val_samples = self.task.get_train_rnn()  
        avg_score = evaluator.batch_reward(current_prompt, val_samples)
        if self.last_sampled_params is not None:
            self.struct_reward_cache[tuple(self.last_sampled_params)] = avg_score

        logger.info(f"🎯 [PromptTemplate] New prompt score with current structure = {avg_score:.4f}")
        return avg_score

    def update(self, evaluator: PromptEvaluator, current_prompt: str):
        """
        Run one RNN optimization step:
        - Sample new structure parameters
        - Sync content to fit structure
        - Evaluate with reward
        - Compute slot-level attributions
        - Reinforce update
        """

        avg_score = self.get_reward(evaluator, current_prompt)
        # Perform slot-level structure attribution if enabled
        if (self.task.config.rnn_structure_contribution
            and self.controller.iter_count != 0
            and self.controller.iter_count % self.controller.attribution_interval == 0):
            logger.info("🔍 [PromptTemplate] Performing slot-level structure attribution...")
            # Get per-slot rewards by perturbing each slot
            slot_rewards = self._structure_attribution(
                params=self.last_sampled_params,
                evaluator=evaluator,
                val_samples=self.task.get_train_rnn(),
                current_prompt=current_prompt
            )
        else:
            slot_rewards = None

        # Ensure forward pass is done here to calculate log_prob_sum and entropy
        # Perform reinforcement learning update (the actual reinforcement step)
        self.controller.reinforce(self.last_log_prob_sum, avg_score, self.last_entropy, slot_rewards)

    def _sync_semantics(self, current_prompt: str) -> str:
        """
        Use the StructureSyncAction to regenerate a semantically complete prompt
        based on the current structural description and training examples.
        """
        return self.sync_action.do(
            current_prompt=current_prompt,
            template_description=self.render(),
        )

    def _structure_attribution(self, params, evaluator:PromptEvaluator, val_samples, current_prompt):
        """
        Perturb each slot in the parameter vector and measure its effect on reward.
        This gives per-slot attribution scores.
        """
        slot_rewards = []
        for i in range(len(params)):
            perturbed = params.copy()

            # Simple trick: add 1 mod slot size to perturb this slot
            slot_dim = self.controller.get_slot_dim(i)
            perturbed[i] = (perturbed[i] + 1) % slot_dim

            new_prompt = self.update_params(perturbed, current_prompt)
            reward = evaluator.batch_reward(new_prompt, val_samples)
            slot_rewards.append(reward)

        return slot_rewards