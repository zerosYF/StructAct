import json
import random
import re
import string
from typing import List, Dict, Any
from task.base_task import TaskBase
from src.config import SearchConfig
from logger import logger

class MedicalExamTask(TaskBase):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.name = "medical_exam"
        path = "dataset/area/MedQA.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            all_examples: List[Dict] = [json.loads(line) for line in f]

        self.origin_prompt = "Answer the question carefully based on medical knowledge."
        self.answer_format_prompt = "At the end of your answer, please provide the answer in the format <answer>A/B/C/D/E</answer>."

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        self.train_size = 100
        self.test_size = 200
        self.train_mcts_size = 80
        self.val_mcts_size = 20
        self._split_data(all_examples)

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        return (
            current_prompt 
            + f"\n\n Question: \n{input}\n" 
            + self.answer_format_prompt
        )

    def extract_tuple(self, sample: dict) -> tuple:
        question = sample.get("question")
        options = sample.get("options")
        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        combined_text = f"Question: {question}\nOptions: \n{options_text}"
        return combined_text, sample.get("answer_idx")

    def samples2text(self, samples: List[dict]) -> str:
        text_blocks = []
        for s in samples:
            question = s['question']
            answer_idx = s['answer_idx']
            options = s['options']
            option_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            text_blocks.append(f"Question: {question}\nOptions: {option_text}\nAnswer: {answer_idx}")
        return "\n".join(text_blocks)

    def _normalize_answer(self, text: str) -> str:
        if not text:
            return ""
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip().upper()
        else:
            text = text.strip().upper()
        label_match = re.search(r"[A-E]", text)
        return label_match.group(0) if label_match else ""

    def get_reward(self, output: str, target: str) -> float:
        norm_out = self._normalize_answer(output)
        norm_gold = self._normalize_answer(target)
        # logger.info(
        #     f"[Reward Evaluation]\n"
        #     f"  Model Answer: {norm_out}\n"
        #     f"  Gold Answer : {norm_gold}"
        # )
        return 1.0 if norm_out == norm_gold else 0.0