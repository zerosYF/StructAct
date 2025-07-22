import random
from typing import Dict, List
from task.base_task import TaskBase
from search.config import SearchConfig
import json
from logger import logger

number_to_word_dict = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21
}

class ObjectCountingTask(TaskBase):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        path = "dataset/BBH/object_counting.json"
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
            sample = {
                "question": input_text,
                "answer": gold
            }
            all_examples.append(sample)

        logger.info(f"✅ [BBH Dataset] Number of samples: {len(all_examples)}")

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        split = int(len(all_examples) * config.split_ratio)
        train_val_data = all_examples[:split]
        self.test_data = all_examples[split:]

        split_1 = int(len(train_val_data) * config.split_ratio_train_val)
        full_train_data = train_val_data[:split_1]
        self.val_data = train_val_data[split_1:]

        split_2 = int(len(full_train_data) * config.split_ratio_train)
        self.train_data_mcts = full_train_data[:split_2]
        self.train_data_rnn = full_train_data[split_2:]

        self.system_prompt = "you are a helpful assistant. Answer the question based on the provided context."

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        """Injects the input question into the current prompt for evaluation."""
        return current_prompt +"\nAnswer only 'yes' or 'no'.\n" + f"\n\nQuestion: {input}"

    def extract_origin_prompt(self) -> str:
        """Returns the original task prompt description."""
        return self.origin_prompt

    def extract_tuple(self, sample) -> tuple:
        """Extracts question and answer tuple from a sample."""
        return sample.get("question"), sample.get("answer")

    def samples2text(self, samples: List[dict]) -> str:
        """Converts a list of samples to a text block of Q&A pairs."""
        return "\n".join([f"Q: {s['question']}\nA: {s['answer']}" for s in samples])