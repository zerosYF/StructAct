import json
import random
from typing import List, Dict
from loguru import logger
from task.base_task import TaskBase

class HateSpeechDetectionTask(TaskBase):
    def __init__(self, config):
        super().__init__(config)
        self.name = "hate_speech_detection"
        path = "dataset/mmlu/ethos.jsonl"

        all_examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line.strip())
                all_examples.append({
                    "text": ex["text"],
                    "label": ex["label_text"].strip().lower()  # "hate speech" or "no hate speech"
                })

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

        self.system_prompt = "You are a helpful assistant. Determine if a given text contains hate speech or not. Respond with 'hate speech' or 'no hate speech' only."

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        return current_prompt + "\nRespond with either 'hate speech' or 'no hate speech'.\n" + f"\n\nText: {input}\nLabel:\n"

    def extract_origin_prompt(self) -> str:
        return "Determine whether a given social media comment contains hate speech. Respond with 'hate speech' or 'no hate speech'."

    def extract_tuple(self, sample) -> tuple:
        return sample["text"], sample["label"]

    def samples2text(self, samples: List[dict]) -> str:
        return "\n".join([f"Text: {s['text']}\nLabel: {s['label']}" for s in samples])

    def _normalize_answer(self, text):
        return text.strip().lower()

    def get_reward(self, output: str, target: str) -> float:
        output = self._normalize_answer(output)
        target = self._normalize_answer(target)
        return 1.0 if output == target else 0.0