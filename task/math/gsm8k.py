import json
import random
from typing import List, Dict
from task.base_task import TaskBase
from loguru import logger
import re

class GSM8KTask(TaskBase):
    def __init__(self, config):
        super().__init__(config)
        self.name = "gsm8k"
        path = "dataset/GSM8K/gsm8k.json"  # 你的文件路径

        with open(path, "r", encoding="utf-8") as f:
            data: List[Dict] = json.load(f)

        all_examples = []
        for ex in data:
            question = ex["text"]
            answer = str(ex["label"]).strip()
            all_examples.append({
                "question": question,
                "answer": answer
            })

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        self.train_size = 500
        self.test_size = 500
        self.train_mcts_size = 400
        self.val_mcts_size = 100
        self._split_data(all_examples)

        self.origin_prompt = "Solve grade-school level math problems using numeric reasoning."
        self.system_prompt = "Solve the math problem and output only the final answer as a number."

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        return current_prompt + f"\n\n {input}\n" + self.answer_format_prompt

    def extract_tuple(self, sample) -> tuple:
        return sample["question"], sample["answer"]

    def samples2text(self, samples: List[dict]) -> str:
        return "\n".join([f"Input: {s['question']}\nOutput: {s['answer']}" for s in samples])
    
    def _normalize_answer(self, text):
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        return text.strip()

    def get_reward(self, output: str, target: str) -> float:
        output = self._normalize_answer(output)
        target = self._normalize_answer(target)
        logger.info(
                f"[Reward Evaluation]\n"
                f"  Model Answer: {output}\n"
                f"  Gold Answer : {target}"
            )
        return 1.0 if output == target else 0.0