import random
from typing import Dict, List
from task.base_task import TaskBase
from search.config import SearchConfig
import json
import re
from logger import logger

class PenguinsTableTask(TaskBase):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.name = "penguins_table"
        path = "dataset/BBH/penguins_in_a_table.json"
        with open(path, "r", encoding="utf-8") as f:
            data: Dict = json.load(f)
        
        self.origin_prompt = data.get("description", "")
        self.name = data.get("name", "unknown_task")
        self.task_prefix = data.get("task_prefix", "")

        all_examples = []
        for ex in data["examples"]:
            input_text = ex["input"]
            target_scores = ex["target_scores"]
            # Select the answer with the highest score
            gold = max(target_scores.items(), key=lambda x: x[1])[0]
            option_text = "\n".join([f"{k}" for k in target_scores.keys()])
            sample = {
                "question": f"Question: {input_text}\nOptions:\n{option_text}",
                "answer": gold
            }
            all_examples.append(sample)

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        self.train_size = 70
        self.test_size = 79
        self.train_mcts_size = 42
        self.val_mcts_size = 14
        self.rl_rnn_size = 14
        self._split_data(all_examples)

        self.system_prompt = "Answer the question based on the provided context."

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        """Injects the input question into the current prompt for evaluation."""
        return (
            current_prompt + "\n"
            + self.task_prefix
            + f"\n\n{input}\n"
            + self.answer_format_prompt
        )

    def extract_tuple(self, sample) -> tuple:
        """Extracts question and answer tuple from a sample."""
        return sample.get("question"), sample.get("answer")

    def samples2text(self, samples: List[dict]) -> str:
        """Converts a list of samples to a text block of Q&A pairs."""
        return "\n".join([f"Input: {s['question']}\nOutput: {s['answer']}" for s in samples])
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize by lowercasing and trimming whitespace."""
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        return text.strip().lower()
    
    def get_reward(self, output: str, target: str) -> float:
        norm_out = self._normalize_answer(output)
        norm_tgt = self._normalize_answer(target)
        logger.info(
                f"[Reward Evaluation]\n"
                f"  Model Answer: {norm_out}\n"
                f"  Gold Answer : {norm_tgt}"
            )
        return 1.0 if norm_out == norm_tgt else 0.0