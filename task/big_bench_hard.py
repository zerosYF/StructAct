import random
from typing import Dict, List
from task.base_task import TaskBase
from search.config import SearchConfig
import json
from logger import logger

class BBHTask(TaskBase):
    def __init__(self, config:SearchConfig):
        super().__init__(config)
        path = "dataset/BBH/epistemic.json"
        with open(path, "r", encoding="utf-8") as f:
            data:Dict = json.load(f)
        self.origin_prompt = data.get("description","")
        self.name = data.get("name", "unknown_task")
        all_examples = []
        for ex in data["examples"]:
            input_text = ex["input"]
            target_scores = ex["target_scores"]
            gold = max(target_scores.items(), key=lambda x: x[1])[0]  # 取得分最高答案
            sample = {
                "question": input_text,
                "answer": gold
            }
            all_examples.append(sample)
        logger.info(f"✅ [BBH数据集] 样本容量: {len(all_examples)}")
        random.seed(config.shuffle_seed)
        random.shuffle(all_examples)
        split = int(len(all_examples) * config.split_ratio)
        self.train_data = all_examples[:split]
        self.test_data = all_examples[split:]
        
        full_train_data = self.train_data  # 备份原始数据
        split_1 = int(len(full_train_data) * config.split_ratio_)

        self.train_data = full_train_data[:split_1]
        self.val_data = full_train_data[split_1:]

        self.system_prompt = "你是一个拥有广博知识的答题人，请回答问题。"

    def inject_final_input(self, current_prompt:str, input:str):
        return current_prompt + f"\n\nquestion:{input}"
    
    def extract_origin_prompt(self):
        return self.origin_prompt
    
    def extract_tuple(self, sample):
        return sample.get("question"), sample.get("answer")
    
    def samples2text(self, samples):
        return "\n".join([f"Q: {s['question']}\nA: {s['answer']}" for s in samples])

    
    
