import random
from typing import Dict, List
from task.base_task import TaskBase
from search.config import SearchConfig
import json
import re
from logger import logger

class GeometricShapesTask(TaskBase):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.name = "geometric_shapes"
        path = "dataset/BBH/geometric_shapes.json"
        with open(path, "r", encoding="utf-8") as f:
            data: Dict = json.load(f)
        
        self.origin_prompt = data.get("description", "")
        self.name = data.get("name", "unknown_task")
        self.system_prompt = "you are a helpful assistant. Answer the question based on the provided context."

        all_examples = []
        for ex in data["examples"]:
            input_text = ex["input"]
            target_scores = ex["target_scores"]
            # Select the answer with the highest score
            gold = max(target_scores.items(), key=lambda x: x[1])[0]
            choices = list(target_scores.keys())
            option_text = "\n".join([f"{k}" for k in choices])
            sample = {
                "question": f"Question: {input_text}\nOptions:\n{option_text}",
                "answer": gold,
                "choices": choices,
            }
            all_examples.append(sample)

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        split = int(len(all_examples) * config.split_ratio)
        train_val_data = all_examples[:150]
        self.test_data = all_examples[150:]

        split_1 = int(len(train_val_data) * config.split_ratio_train_val)
        full_train_data = train_val_data[:split_1]
        self.val_data = train_val_data[split_1:]

        split_2 = int(len(full_train_data) * config.split_ratio_train)
        self.train_data_mcts = full_train_data[:split_2]
        self.train_data_rnn = full_train_data[split_2:]


    def inject_final_input(self, current_prompt: str, input: str) -> str:
        """Injects the input question into the current prompt for evaluation."""
        return current_prompt +"\nOnly output one in options as anwser\n" + f"\n\nQuestion: {input}\n Anwser:\n"

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
        """Normalize text by lowercasing, stripping, and removing punctuation."""
        text = text.strip().lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text
    
    def get_reward(self, output: str, target: str) -> float:
        """Compares normalized output and target answer for reward calculation."""
        norm_out = self._normalize_answer(output)
        norm_gold = self._normalize_answer(target)
        return 1.0 if norm_out == norm_gold else 0.0