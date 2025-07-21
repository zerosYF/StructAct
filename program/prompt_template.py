from typing import List
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
            lr=config.rnn_lr
        )
        self.task = task
        self.sync_action = StructureSyncAction(task, self.task.extract_origin_prompt())
        self.last_sampled_params = None
        self.count_max = config.action_structure_flush_ratio
        self.count = 1

    def _get_search_space(self) -> List[int]:
        """
        Gather the full search space across all blocks.
        """
        search_space = []
        for block in self.blocks:
            search_space.extend(block.get_search_space())
        return search_space

    def render(self) -> str:
        """
        Render each block as a JSON-like string and join them.
        """
        return "\n".join([f"{block.render()}" for block in self.blocks])

    def describe(self) -> str:
        """
        Generate a structured natural language description of the full template.
        """
        header = (
            "This is a structured prompt template composed of the following functional blocks. \n"
            "Information in “ ” is most important you should focus on.\n"
            "Each block plays a distinct role in guiding the model’s behavior:\n"
        )
        block_descriptions = "\n".join([f"- {block.describe()}" for block in self.blocks])
        return header + block_descriptions

    def update_by_controller(self, evaluator: PromptEvaluator, current_prompt: str):
        """
        Run one RNN optimization step:
        - Sample new structure parameters
        - Sync content to fit structure
        - Evaluate with reward
        - Compute slot-level attributions
        - Reinforce update
        """
        if self.count != self.count_max:
            self.count += 1
            return current_prompt
        self.count = 1
        # Step 1: Sample structure parameters from the controller
        flat_params, log_prob_sum, entropy = self.controller.train_step()

        if flat_params == self.last_sampled_params:
            logger.warning("Repeated parameters detected, skipping update.")
        else:
            self.last_sampled_params = flat_params
            # Step 2: Assign parameters to each block
            idx = 0
            for block in self.blocks:
                num = block.get_num_slots()
                block.set_hyperparams(flat_params[idx:idx + num])
                idx += num

            # Step 3: Sync semantic content to match the new structure
            current_prompt = self._sync_semantics(current_prompt)
        ## Step 4: Evaluate the new prompt using the evaluator
        val_samples = self.task.sample_train_rnn()  # Sample a subset of the training set
        total_score = sum(evaluator.batch_reward(current_prompt, val_samples))
        avg_score = total_score / len(val_samples)
        logger.info(f"🎯 [PromptTemplate] New prompt score with current structure = {avg_score:.4f}")

        # Step 4.5: Perform slot-level structure attribution if enabled
        if self.task.config.rnn_structure_contribution:
            logger.info("🔍 [PromptTemplate] Performing slot-level structure attribution...")
            # Get per-slot rewards by perturbing each slot
            slot_rewards = self._structure_attribution(
                params=flat_params,
                evaluator=evaluator,
                val_samples=val_samples,
                current_prompt=current_prompt
            )
        else:
            slot_rewards = None

        # Step 5: Perform forward pass to compute log_prob_sum and entropy
        # Ensure forward pass is done here to calculate log_prob_sum and entropy
        # Perform reinforcement learning update (the actual reinforcement step)
        self.controller.reinforce(log_prob_sum, avg_score, entropy, slot_rewards)

        return current_prompt

    def _sync_semantics(self, current_prompt: str) -> str:
        """
        Use the StructureSyncAction to regenerate a semantically complete prompt
        based on the current structural description and training examples.
        """
        template_description = self.describe()
        return self.sync_action.do(
            current_prompt=current_prompt,
            template_description=template_description,
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

            # Apply perturbed parameters to blocks
            idx = 0
            for block in self.blocks:
                num = block.get_num_slots()
                block.set_hyperparams(perturbed[idx:idx + num])
                idx += num

            # Regenerate prompt and evaluate
            new_prompt = self._sync_semantics(current_prompt)
            reward = sum(evaluator.batch_reward(new_prompt, val_samples)) / len(val_samples)
            slot_rewards.append(reward)

        return slot_rewards