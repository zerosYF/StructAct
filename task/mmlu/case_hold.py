import json
import random
import re
from typing import List, Dict
from logger import logger
from task.base_task import TaskBase
from search.config import SearchConfig


class LegalHoldingTask(TaskBase):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.name = "legal_holding"
        path = "dataset/area/CaseHold.jsonl"   # 你的实际路径
        with open(path, "r", encoding="utf-8") as f:
            all_examples: List[Dict] = [json.loads(line) for line in f]

        self.origin_prompt = (
            "You are given a legal case snippet with a <HOLDING> placeholder. "
            "Choose the single correct holding from the candidate options. "
        )
        self.answer_format_prompt = "At the end show the index of the correct option, wrapped in <answer>...</answer>."

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        self.train_size = 100
        self.test_size = 200
        self.train_mcts_size = 80
        self.val_mcts_size = 20
        self._split_data(all_examples)

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        case_text, options = input
        options_str = "\n".join([f"{i}. {opt}" for i, opt in enumerate(options)])
        return (
            current_prompt
            + f"\n\nCase:\n{case_text}\n\nOptions:\n{options_str}\n"
            + self.answer_format_prompt
        )

    def extract_tuple(self, sample: dict) -> tuple:
        case_text = sample["0"]
        options = [sample[str(i)] for i in range(1, 6) if str(i) in sample]
        gold_idx = int(sample["11"])   
        return (case_text, options), gold_idx

    def samples2text(self, samples: List[dict]) -> str:
        texts = []
        for s in samples:
            (case_text, options), gold_idx = self.extract_tuple(s)
            opts_str = "\n".join([f"{i}. {opt}" for i, opt in enumerate(options)])
            texts.append(f"Case: {case_text}\nOptions:\n{opts_str}\nAnswer: {gold_idx}")
        return "\n\n".join(texts)

    def _normalize_answer(self, text: str) -> int:
        if isinstance(text, int):
            return text
        if not text:
            return -1
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if not match:
            return -1
        ans = match.group(1).strip()
        try:
            return int(ans)
        except ValueError:
            return -1

    def get_reward(self, output: str, target: int) -> float:
        pred_idx = self._normalize_answer(output)
        gold_idx = int(target)
        # logger.info(
        #     f"[Reward Evaluation]\n"
        #     f"  Predicted Index: {pred_idx}\n"
        #     f"  Gold Index     : {gold_idx}"
        # )
        return 1.0 if pred_idx == gold_idx else 0.0