import json
import random
from typing import List, Dict
from loguru import logger
from task.base_task import TaskBase 

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
                answer = ex["options"][ex["answer_index"]].strip()
                all_examples.append({
                    "question": question,
                    "answer": answer,
                })

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        split = int(len(all_examples) * config.split_ratio)
        full_train_data = all_examples[:100]
        self.train_data_all = full_train_data
        self.test_data = all_examples[100:200]

        split_1 = int(len(full_train_data) * config.split_ratio_)
        self.train_data_mcts = full_train_data[:split_1]
        self.eval_data_mcts = full_train_data[split_1:]

        self.system_prompt = (
            "You are a helpful business assistant. Choose the most appropriate option for the question below. "
            "Only output the content of the correct option, not the letter."
        )

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        return current_prompt + "\n\n" + input + "\nAnswer:\n"

    def extract_origin_prompt(self) -> str:
        return "Answer business and management multiple-choice questions by selecting the correct option content."

    def extract_tuple(self, sample) -> tuple:
        return sample["question"], sample["answer"]

    def samples2text(self, samples: List[dict]) -> str:
        return "\n".join([f"{s['question']}\nAnswer: {s['answer']}" for s in samples])

    def format_question(self, question: str, options: List[str]) -> str:
        return f"Question: {question}\n" + "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])

    def _normalize_answer(self, text: str) -> str:
        return text.strip().lower()

    def get_reward(self, output: str, target: str) -> float:
        return 1.0 if self._normalize_answer(output) == self._normalize_answer(target) else 0.0