from typing import Dict, List, Any
from abc import ABC, abstractmethod
from search.config import SearchConfig
from model.model import getOptimModel
import random
class TaskBase(ABC):
    def __init__(self, config:SearchConfig):
        self.name = "BaseTask"
        self.config = config
        # dataset split
        self.train_size:int = 0
        self.test_size:int = 0
        self.train_mcts_size:int = 0
        self.val_mcts_size:int = 0
        
        self.train_data_mcts = None
        self.eval_data_mcts = None
        self.test_data = None
        self.system_prompt:str = None
        self.origin_prompt:str = None
        self.answer_format_prompt:str = "At the end show the answer option bracketed between <answer> and </answer>."
    
    def _split_data(self, all_examples:list):
        assert self.train_size == self.train_mcts_size + self.val_mcts_size

        full_train_data = all_examples[:self.train_size]
        self.test_data = all_examples[self.train_size:self.train_size + self.test_size]

        self.train_data_mcts = full_train_data[:self.train_mcts_size]
        self.eval_data_mcts = full_train_data[self.train_mcts_size:self.train_mcts_size + self.val_mcts_size]
    # train mcts actions sample mini_batch
    def sample_train_mcts(self, batch_size:int) -> List[Dict]:
        assert batch_size < self.train_mcts_size
        return random.sample(self.train_data_mcts, batch_size)
    
    # train rnn structure
    def get_train_rnn(self) -> List[Dict]:
        return self.eval_data_mcts 
    
    # get reward to update mcts q
    def get_eval(self) -> List[Dict]:
        return self.eval_data_mcts
    
    def get_test(self) -> List[Dict]:
        return self.test_data
    
    def process_origin_prompt(self, fs=False, cot=False, fs_cot=False) -> str:
        prompt_parts = [self.origin_prompt.strip()]

        if fs:
            sample_lst = self.sample_train_mcts(3)
            fewshot_text = self.samples2text(sample_lst)
            if fs_cot:
                system_prompt = (
                    "You are a prompt converter. Your task is to convert multiple question-answer pairs "
                    "into few-shot examples that include explanation and final answer. "
                    "For each example, follow this format:\n\n"
                    "Question: <original question>\n"
                    "Explanation: <step-by-step reasoning>\n"
                    "Answer: <final answer> \n\n"
                    "Do not add any extra commentary or formatting. Keep the input question unchanged."
                )
                fewshot_text = getOptimModel().api_call(fewshot_text)

            prompt_parts.append("Here are some examples:\n" + fewshot_text)

        if cot:
            format_instruction = (
                "Please think step by step, and explain your reasoning before choosing an answer. "
                "Respond in the following format:\n"
                "<explanation> ... </explanation>\n"
                "<answer> ... </answer>"
            )
        else:
            format_instruction = (
                "Select the best answer based on the question and options. "
                "Respond in the following format without other text:\n"
                "<answer> ... </answer>"
            )

        prompt_parts.append(format_instruction)
        return "\n".join(prompt_parts)
    
    @abstractmethod
    def inject_final_input(self, current_prompt:str, input:str) -> str:
        pass

    @abstractmethod
    def extract_tuple(self, sample:dict) -> tuple:
        pass

    @abstractmethod
    def samples2text(self, samples:List[Dict]) -> str:
        pass

    @abstractmethod
    def _normalize_answer(self, text: str) -> str:
        """Normalize text by lowercasing, stripping, and removing punctuation."""
        pass

    @abstractmethod
    def get_reward(self, output: Any, target: Any) -> float:
        pass