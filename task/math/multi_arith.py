import json
import random
from typing import List, Dict
from task.base_task import TaskBase
from loguru import logger

class SimpleMathReasoningTask(TaskBase):
    def __init__(self, config):
        super().__init__(config)
        self.name = "simple_math_reasoning"
        path = "dataset/GSM8K/multi_arith.json"  # 你存放数据的路径

        with open(path, "r", encoding="utf-8") as f:
            data: List[Dict] = json.load(f)

        all_examples = []
        for ex in data:
            question = ex["question"].strip()
            answer = str(ex["final_ans"]).strip()
            all_examples.append({
                "question": question,
                "answer": answer
            })

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        split = int(len(all_examples) * config.split_ratio)
        full_train_data = all_examples[:split]
        self.train_data_all = full_train_data
        self.test_data = all_examples[split:]

        split_1 = int(len(full_train_data) * config.split_ratio_)
        self.train_data_mcts = full_train_data[:split_1]
        self.eval_data_mcts = full_train_data[split_1:]

        self.system_prompt = (
            "You are a helpful assistant. Please solve the math word problem carefully and return only the final numeric answer."
        )

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        return current_prompt + "\nOnly output the final numeric answer, without explanation.\n" + f"\n\nQuestion: {input}\nAnswer:\n"

    def extract_origin_prompt(self) -> str:
        return "Solve simple arithmetic word problems with numeric answers."

    def extract_tuple(self, sample) -> tuple:
        return sample["question"], sample["answer"]

    def samples2text(self, samples: List[dict]) -> str:
        return "\n".join([f"Q: {s['question']}\nA: {s['answer']}" for s in samples])
    
    def _normalize_answer(self, text):
        return text.strip()

    def get_reward(self, output: str, target: str) -> float:
        output = self._normalize_answer(output)
        target = self._normalize_answer(target)
        return 1.0 if output == target else 0.0