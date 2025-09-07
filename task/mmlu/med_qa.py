import json
import random
import re
import string
from typing import List, Dict, Any
from task.base_task import TaskBase
from search.config import SearchConfig
from logger import logger

class MedicalExamTask(TaskBase):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.name = "medical_exam"
        path = "dataset/area/MedQA.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            all_examples: List[Dict] = [json.loads(line) for line in f]

        self.system_prompt = "Answer the USMLE-style question carefully based on medical knowledge."
        self.origin_prompt = (
            "You are given a USMLE-style multiple-choice medical question. "
            "Choose the single best answer. "
            "and always enclose the final choice in <answer>…</answer> tags."
        )

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        self.train_size = 100
        self.test_size = 200
        self.train_mcts_size = 80
        self.val_mcts_size = 20
        self._split_data(all_examples)

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        return current_prompt + f"\n\n{input}\n" + self.answer_format_prompt

    def extract_tuple(self, sample: dict) -> tuple:
        return sample.get("question"), sample.get("answer_idx")  # 直接返回标签

    def samples2text(self, samples: List[dict]) -> str:
        """few-shot 样例显示题目和选项标签"""
        text_blocks = []
        for s in samples:
            text_blocks.append(f"Question: {s['question']}\nAnswer: <answer>{s['answer_idx']}</answer>")
        return "\n".join(text_blocks)

    def _normalize_answer(self, text: str) -> str:
        """
        提取 <answer> 标签中的内容，再抓取首个合法选项 A/B/C/D/E
        """
        if not text:
            return ""
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip().upper()
        else:
            text = text.strip().upper()
        # 提取首个选项标签 A/B/C/D/E
        label_match = re.search(r"[A-E]", text)
        return label_match.group(0) if label_match else ""

    def get_reward(self, output: str, target: str) -> float:
        """只比较选项标签是否一致"""
        norm_out = self._normalize_answer(output)
        norm_gold = self._normalize_answer(target)
        logger.info(
            f"[Reward Evaluation]\n"
            f"  Model Answer: {norm_out}\n"
            f"  Gold Answer : {norm_gold}"
        )
        return 1.0 if norm_out == norm_gold else 0.0