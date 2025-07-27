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

        split = int(len(all_examples) * config.split_ratio)
        full_train_data = all_examples[:70]
        self.train_data_all = full_train_data
        self.test_data = all_examples[70:]

        split_1 = int(len(full_train_data) * config.split_ratio_)
        self.train_data_mcts = full_train_data[:split_1]
        self.eval_data_mcts = full_train_data[split_1:]

        self.system_prompt = "you are a helpful assistant. Answer the question based on the provided context."

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        """Injects the input question into the current prompt for evaluation."""
        return (
            current_prompt
            + "\n"
            + self.task_prefix
            + f"\n\nQuestion: {input}\n"
            + self.answer_format_prompt
        )

    def extract_origin_prompt(self) -> str:
        """Returns the original task prompt description."""
        return self.origin_prompt

    def extract_tuple(self, sample) -> tuple:
        """Extracts question and answer tuple from a sample."""
        return sample.get("question"), sample.get("answer")

    def samples2text(self, samples: List[dict]) -> str:
        """Converts a list of samples to a text block of Q&A pairs."""
        return "\n".join([f"Q: {s['question']}\nA: {s['answer']}" for s in samples])
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize by lowercasing and trimming whitespace."""
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        return text.strip().lower()
    
    def get_reward(self, output: str, target: str) -> float:
        norm_out = self._normalize_answer(output)
        norm_tgt = self._normalize_answer(target)
        return 1.0 if norm_out == norm_tgt else 0.0