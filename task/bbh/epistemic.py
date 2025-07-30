import random
from typing import Dict, List
from task.base_task import TaskBase
from search.config import SearchConfig
import json
import re
from logger import logger

class EpistemicTask(TaskBase):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.name = "epistemic"
        path = "dataset/BBH/epistemic.json"
        with open(path, "r", encoding="utf-8") as f:
            data: Dict = json.load(f)
        
        self.origin_prompt = data.get("description", "")
        self.name = data.get("name", "unknown_task")
        self.task_prefix = data.get("task_prefix", "")
        self.system_prompt = "Answer the question based on the provided context."

        all_examples = []
        for ex in data["examples"]:
            input_text = ex["input"]
            target_scores = ex["target_scores"]
            # Select the answer with the highest score
            gold = max(target_scores.items(), key=lambda x: x[1])[0]
            choices = list(target_scores.keys())
            option_text = "\n".join([f"{k}" for k in choices])
            sample = {
                "question": f"Question: {input_text}\nOptions:\n{option_text}",
                "answer": gold,
                "choices": choices,
            }
            all_examples.append(sample)

        logger.info(f"✅ [{self.name} Dataset] Number of samples: {len(all_examples)}")

        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)

        self.train_size = 500
        self.test_size = 500
        self.train_mcts_size = 300
        self.val_mcts_size = 100
        self.rl_rnn_size = 100
        self._split_data(all_examples)

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        """Injects the input question into the current prompt for evaluation."""
        return current_prompt  + f"\n{self.task_prefix}" + f"\n\n{input}\n" + self.answer_format_prompt

    def extract_tuple(self, sample) -> tuple:
        """Extracts question and answer tuple from a sample."""
        return sample.get("question"), sample.get("answer")

    def samples2text(self, samples: List[dict]) -> str:
        """Converts a list of samples to a text block of Q&A pairs."""
        return "\n".join([f"Input: {s['question']}\nOutput: {s['answer']}" for s in samples])
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize text by lowercasing, stripping, and removing punctuation."""
        match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        text = text.strip().lower()
        text = re.sub(r"[^\x20-\x7E]", "", text)
        return text
    
    def get_reward(self, output: str, target: str) -> float:
        """Compares normalized output and target answer for reward calculation."""
        # logger.info(
        #         f"  Model Output: {output}\n"
        #     )
        norm_out = self._normalize_answer(output)
        norm_gold = self._normalize_answer(target)
        logger.info(
                f"[Reward Evaluation]\n"
                f"  Model Answer: {norm_out}\n"
                f"  Gold Answer : {norm_gold}"
            )
        return 1.0 if norm_out == target else 0.0

    
    
