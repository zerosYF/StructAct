import json
import random
from typing import List, Dict
from loguru import logger
from task.base_task import TaskBase  

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
        self.test_data = all_examples[100:300]

        split_1 = int(len(full_train_data) * config.split_ratio_)
        self.train_data_mcts = full_train_data[:50]
        self.eval_data_mcts = full_train_data[50:]

        self.system_prompt = (
            "You are a helpful engineering assistant. Choose the most accurate answer to the technical question below. "
            "Only output the final answer content, not the option letter."
        )

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        return current_prompt + "\n\n" + input + "\nAnswer:\n"

    def extract_origin_prompt(self) -> str:
        return "Answer electrical and electronics engineering multiple-choice questions."

    def extract_tuple(self, sample) -> tuple:
        return sample["question"], sample["answer"]

    def samples2text(self, samples: List[dict]) -> str:
        return "\n".join([f"{s['question']}\nAnswer: {s['answer']}" for s in samples])

    def format_question(self, question: str, options: List[str]) -> str:
        return f"Question: {question}\n" + "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options) if opt != "N/A"])

    def _normalize_answer(self, text: str) -> str:
        return text.strip().lower()

    def get_reward(self, output: str, target: str) -> float:
        return 1.0 if self._normalize_answer(output) == self._normalize_answer(target) else 0.0