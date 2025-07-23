import random
from typing import Dict, List
from task.base_task import TaskBase
from search.config import SearchConfig
import json
from logger import logger

class PenguinsTableTask(TaskBase):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        path = "dataset/BBH/penguins_in_a_table.json"
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
                "question": f"Question: {input_text}\nOptions:\n{option_text}",
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
    
    def get_reward(self, output:str, target:str) -> float:
        return 1.0 if output.strip().lower() == target.strip().lower() else 0.0