import json
import random
from typing import List, Dict
from loguru import logger
from task.base_task import TaskBase

class HateSpeechDetectionTask(TaskBase):
    def __init__(self, config):
        super().__init__(config)
        self.name = "multilabel_hatespeech"
        path = "dataset/mmlu/ethos_ml.jsonl"  # 根据实际路径修改

        all_examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line.strip())
                text = ex["text"]
                labels = {
                    "violence": ex["violence"],
                    "directed_vs_generalized": ex["directed_vs_generalized"],
                    "gender": ex["gender"],
                    "race": ex["race"],
                    "national_origin": ex["national_origin"],
                    "disability": ex["disability"],
                    "religion": ex["religion"],
                    "sexual_orientation": ex["sexual_orientation"]
                }
                all_examples.append({
                    "question": text,
                    "answer": labels
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
            "You are a hate speech detection system. For a given text, "
            "identify whether each of the following labels applies: "
            "violence, directed_vs_generalized, gender, race, national_origin, "
            "disability, religion, sexual_orientation. "
            "Output a JSON object with all 8 fields as binary (0 or 1)."
        )

    def inject_final_input(self, current_prompt: str, input: str) -> str:
        return (
            current_prompt +
            "\n\nText: " + input +
            "\nAnswer with a raw JSON dictionary, without markdown formatting or code block:\n"
        )

    def extract_origin_prompt(self) -> str:
        return "Classify text into multiple hate speech labels."

    def extract_tuple(self, sample: Dict) -> tuple:
        return sample["question"], sample["answer"]

    def samples2text(self, samples: List[dict]) -> str:
        return "\n".join([
            f"Text: {s['question']}\nLabels: {json.dumps(s['answer'])}"
            for s in samples
        ])

    def _normalize_answer(self, output: str) -> Dict:
        try:
            return json.loads(output)
        except Exception:
            return {}

    def get_reward(self, output: str, target: Dict) -> float:
        pred = self._normalize_answer(output)
        if not isinstance(pred, dict):
            return 0.0
        return 1.0 if pred == target else 0.0