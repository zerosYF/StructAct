import json
import random
from typing import List, Dict
from loguru import logger
from task.base_task import TaskBase  
import re

class EngineeringMCQTask(TaskBase):
    def __init__(self, config):
        super().__init__(config)
        self.name = "engineering_mcq"
        path = "dataset/mmlu/engineering.jsonl"

        all_examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line.strip())
                question = self.format_question(ex["question"], ex["options"])
                answer = chr(65 + ex["answer_index"])
                all_examples.append({
                    "question": question,
                    "answer": answer,
                })

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        self.train_size = 100
        self.test_size = 200
        self.train_mcts_size = 80
        self.val_mcts_size = 20
        self._split_data(all_examples)

        self.origin_prompt = "Answer electrical and electronics engineering multiple-choice questions."
        self.answer_format_prompt = "At the end of your answer, please provide the final answer in the format <answer>A</answer>, where A is one of A, B, C, or D."

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        return (
            current_prompt 
            + f"\n\n{input}\n" 
            + self.answer_format_prompt
        )
    
    def extract_tuple(self, sample) -> tuple:
        return sample["question"], sample["answer"]

    def samples2text(self, samples: List[dict]) -> str:
        return "\n".join([f"{s['question']}\nAnswer: {s['answer']}" for s in samples])

    def format_question(self, question: str, options: List[str]) -> str:
        return f"Question: {question}\n" + "Options:\n" +"\n".join([
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options) if opt != "N/A"
        ])

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
        # logger.info(
        #         f"[Reward Evaluation]\n"
        #         f"  Model Answer: {norm_out}\n"
        #         f"  Gold Answer : {norm_gold}"
        #     )
        return 1.0 if norm_out == norm_gold else 0.0