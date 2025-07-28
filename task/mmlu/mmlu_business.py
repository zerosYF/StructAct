import json
import random
from typing import List, Dict
from loguru import logger
from task.base_task import TaskBase 
import re

class BusinessMCQTask(TaskBase):
    def __init__(self, config):
        super().__init__(config)
        self.name = "business_mcq"
        path = "dataset/mmlu/business.jsonl"

        all_examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line.strip())
                question = self.format_question(ex["question"], ex["options"])
                letter_answer = chr(65 + ex["answer_index"])  # A, B, C...
                all_examples.append({
                    "question": question,
                    "answer": letter_answer,
                })

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        split = int(len(all_examples) * config.split_ratio)
        full_train_data = all_examples[:100]
        self.test_data = all_examples[100:300]

        split_1 = int(len(full_train_data) * config.split_ratio_)
        self.train_data_mcts = full_train_data[:50]
        full_reward_acc = full_train_data[50:]

        split_2 = int(len(full_reward_acc) * config.split_ratio__)
        self.eval_data_mcts = full_reward_acc[:split_2]
        self.train_data_rnn = full_reward_acc[split_2:]

        self.system_prompt = (
            "You are a helpful business assistant. Choose the most appropriate option (A, B, C, D) for the question below. "
            "Only output the letter of the correct answer (e.g., 'A')."
        )

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        return current_prompt + f"\n\nQuestion: {input}\nAnswer:"

    def extract_origin_prompt(self) -> str:
        return "Answer business and management multiple-choice questions by selecting the correct option letter."

    def extract_tuple(self, sample) -> tuple:
        return sample["question"], sample["answer"]

    def samples2text(self, samples: List[dict]) -> str:
        return "\n".join([f"{s['question']}\nAnswer: {s['answer']}" for s in samples])

    def format_question(self, question: str, options: List[str]) -> str:
        return f"Question: {question}\n" + "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])

    def _normalize_answer(self, text: str) -> str:
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if match:
            text = match.group(1)

        text = text.strip()

        match = re.search(r"\b([A-D])\b", text.upper())
        if match:
            return match.group(1).upper()

        return text[:1].upper()

    def get_reward(self, output: str, target: str) -> float:
        norm_out = self._normalize_answer(output)
        norm_gold = self._normalize_answer(target)
        logger.info(
                f"[Reward Evaluation]\n"
                f"  Model Answer: {norm_out}\n"
                f"  Gold Answer : {norm_gold}"
            )
        return 1.0 if norm_out == norm_gold else 0.0