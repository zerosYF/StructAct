import random
from typing import Dict, List
from task.base_task import TaskBase
from src.config import SearchConfig
import json
import re
from logger import logger

class TemporalSequencesTask(TaskBase):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.name = "temporal_sequences"
        path = "dataset/BBH/temporal_sequences.json"
        with open(path, "r", encoding="utf-8") as f:
            data: Dict = json.load(f)
        
        self.origin_prompt = data.get("description", "")
        self.name = data.get("name", "unknown_task")

        all_examples = []
        for ex in data["examples"]:
            input_text = ex["input"]
            target_scores = ex["target_scores"]
            # Select the answer with the highest score
            gold = max(target_scores.items(), key=lambda x: x[1])[0]
            option_text = "\n".join([f"{k}" for k in target_scores.keys()])
            sample = {
                "question": f"Question: \n{input_text}\nOptions:\n{option_text}",
                "answer": gold
            }
            all_examples.append(sample)

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        self.train_size = 300
        self.test_size = 500
        self.train_mcts_size = 240
        self.val_mcts_size = 60
        self._split_data(all_examples)
        

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        """Injects the input question into the current prompt for evaluation."""
        return (
            current_prompt 
            + f"\n\nInput:\n{input}\n" 
            + self.answer_format_prompt
        )

    def extract_tuple(self, sample) -> tuple:
        """Extracts question and answer tuple from a sample."""
        return sample.get("question"), sample.get("answer")

    def samples2text(self, samples: List[dict]) -> str:
        """Converts a list of samples to a text block of Q&A pairs."""
        return "\n".join([f"Input: \n{s['question']}\nOutput: {s['answer']}" for s in samples])
    
    def _normalize_answer(self, text: str) -> str:
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        return text.strip().lower()
    
    def get_reward(self, output: str, target: str) -> float:
        norm_out = self._normalize_answer(output)
        norm_tgt = self._normalize_answer(target)
        return 1.0 if norm_out == norm_tgt else 0.0