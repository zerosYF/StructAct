import json
import random
import logging
from typing import List, Dict
from task.base_task import TaskBase
from search.config import SearchConfig

logger = logging.getLogger(__name__)

class PrologMathTask(TaskBase):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.name = "prolog_math"
        path = "dataset/GSM8K/gsm_prolog_test.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]

        self.origin_prompt = "Generate a piece of Prolog code to solve the given math problem."
        self.name = "prolog_math"

        all_examples = []
        for ex in data:
            instruction = ex.get("instruction", "")
            input_text = ex.get("input", "")
            output_code = ex.get("output", "")

            question = f"{instruction}\nProblem: {input_text}\nGenerate Prolog code:"
            sample = {
                "question": question,
                "answer": output_code
            }
            all_examples.append(sample)

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

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

        self.system_prompt = "You are a helpful assistant who writes Prolog code to solve math problems."

    def inject_final_input(self, current_prompt: str, input_text: str) -> str:
        return current_prompt + "\nPlease provide the Prolog solution code.\n" + f"\n\n{input_text}\nAnswer:\n"

    def extract_origin_prompt(self) -> str:
        return self.origin_prompt

    def extract_tuple(self, sample) -> tuple:
        return sample.get("question"), sample.get("answer")

    def samples2text(self, samples: List[dict]) -> str:
        return "\n\n".join([f"Q: {s['question']}\nA:\n{s['answer']}" for s in samples])
    
    def _normalize_answer(self, output: str) -> str:
        return output.strip()

    def get_reward(self, output: str, target: str) -> float:
        output = self._normalize_answer(output)
        target = self._normalize_answer(target)
        return 1.0 if output == target else 0.0