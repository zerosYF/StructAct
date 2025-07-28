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
        self.train_data_mcts = 40
        self.val_mcts_size = 29
        self.rl_rnn_size = 31
        self._split_data(all_examples)

        self.origin_prompt = "Answer electrical and electronics engineering multiple-choice questions."
        self.system_prompt = (
            "Choose the most accurate answer to the technical question below. "
            "Only output the option letter (e.g., A, B, C), not the content."
        )

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        return current_prompt + f"\n\nQuestion: {input}\nAnswer:"
    
    def extract_tuple(self, sample) -> tuple:
        return sample["question"], sample["answer"]

    def samples2text(self, samples: List[dict]) -> str:
        return "\n".join([f"{s['question']}\nAnswer: {s['answer']}" for s in samples])

    def format_question(self, question: str, options: List[str]) -> str:
        return f"Question: {question}\n" + "\n".join([
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options) if opt != "N/A"
        ])

    def _normalize_answer(self, text: str) -> str:
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if match:
            text = match.group(1)

        text = re.sub(r"^(Answer\s*[:：]?\s*)?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"[\.\s]+$", "", text.strip())  
        return text.strip().upper()

    def get_reward(self, output: str, target: str) -> float:
        norm_out = self._normalize_answer(output)
        norm_gold = self._normalize_answer(target)
        logger.info(
                f"[Reward Evaluation]\n"
                f"  Model Answer: {norm_out}\n"
                f"  Gold Answer : {norm_gold}"
            )
        return 1.0 if norm_out == norm_gold else 0.0