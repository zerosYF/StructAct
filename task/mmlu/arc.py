import json
import random
import re
from typing import List, Dict
from logger import logger
from task.base_task import TaskBase
from search.config import SearchConfig


class PhysicsMCQTask(TaskBase):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.name = "physics_mcq"
        path = "dataset/area/arc.jsonl"  
        with open(path, "r", encoding="utf-8") as f:
            all_examples: List[Dict] = [json.loads(line) for line in f]

        self.origin_prompt = "Answer the multiple-choice question based on the context."
        self.answer_format_prompt = "At the end of your response, provide your answer in the format: <answer>Your_Chosen_Letter(A, B, C, D, ...)</answer>."

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        self.train_size = 100
        self.test_size = 200
        self.train_mcts_size = 80
        self.val_mcts_size = 20
        self._split_data(all_examples)

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

    def inject_final_input(self, current_prompt: str, input: tuple) -> str:
        question, choices_text, choices_label = input
        options_str = "\n".join([f"{label}. {text}" for label, text in zip(choices_label, choices_text)])
        return current_prompt + f"\n\nQuestion:\n{question}\n\nOptions:\n{options_str}\n" + self.answer_format_prompt

    def extract_tuple(self, sample: dict) -> tuple:
        question = sample["question"]
        choices_text = sample["choices"]["text"]
        choices_label = sample["choices"]["label"]
        gold_letter = sample["answerKey"].upper()
        return (question, choices_text, choices_label), gold_letter

    def samples2text(self, samples: List[dict]) -> str:
        texts = []
        for s in samples:
            (question, choices_text, choices_label), gold_letter = self.extract_tuple(s)
            options_str = "\n".join([f"{label}. {text}" for label, text in zip(choices_label, choices_text)])
            texts.append(f"Question: {question}\nOptions:\n{options_str}\nAnswer: {gold_letter}")
        return "\n\n".join(texts)

    def _normalize_answer(self, text: str) -> str:
        if not text:
            return ""
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if match:
            return match.group(1).strip().upper()
        return ""

    def get_reward(self, output: str, target: str) -> float:
        pred_letter = self._normalize_answer(output)
        gold_letter = target.upper()
        logger.info(
            f"[Reward Evaluation]\n"
            f"  Predicted Letter: {pred_letter}\n"
            f"  Gold Letter     : {gold_letter}"
        )
        return 1.0 if pred_letter == gold_letter else 0.0