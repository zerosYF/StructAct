import json
import random
import re
from typing import List, Dict
from logger import logger
from task.base_task import TaskBase
from src.config import SearchConfig


class TemporalSequenceTask(TaskBase):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.name = "temporal_sequence"
        path = "dataset/BBEH/bbeh_by_task/temporal_sequence.json" 

        with open(path, "r", encoding="utf-8") as f:
            all_examples: List[Dict] = json.load(f)  

        self.origin_prompt = (
            "You are given several participants' weekly schedules and rules. "
        )

        self.answer_format_prompt = (
            "Your task is to compute:\n"
            " - X: the length of the longest possible meeting (minutes)\n"
            " - Y: the number of options for such meetings\n\n"
            "At the end of your response  include<answer>X, Y</answer>."
        )

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        self.train_size = 120
        self.test_size = 80
        self.train_mcts_size = 100
        self.val_mcts_size = 20
        self._split_data(all_examples)

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

    def inject_final_input(self, current_prompt: str, input: tuple) -> str:
        question = input
        return current_prompt + f"\n\nQuestion:\n{question}\n" + self.answer_format_prompt

    def extract_tuple(self, sample: dict) -> tuple:
        question = sample["input"]
        gold = sample["target"].strip()
        return question, gold

    def samples2text(self, samples: List[dict]) -> str:
        texts = []
        for s in samples:
            question, gold = self.extract_tuple(s)
            texts.append(f"Question: {question}\nAnswer: {gold}")
        return "\n\n".join(texts)

    def _normalize_answer(self, text: str) -> str:
        if not text:
            return ""
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def get_reward(self, output: str, target: str) -> float:
        pred = self._normalize_answer(output)
        gold = target.strip()
        logger.info(
            f"[Reward Evaluation]\n"
            f"  Predicted: {pred}\n"
            f"  Gold     : {gold}"
        )
        return 1.0 if pred == gold else 0.0