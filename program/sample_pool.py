import random
from collections import deque
from logger import logger
import math
import numpy as np
from abc import abstractmethod
from enum import Enum

class SampleType(Enum):
    Positive = "positive"
    Negative = "negative"


class PoolSample:
    def __init__(self, raw_sample: dict):
        self.raw = raw_sample  # question/answer/choices
        self.reward_history = []
        self.baseline_reward = 0.0
        self.informative_score = 0.0

    def update(self, is_correct: float):
        """is_correct 可以是 0/1 或 [0,1] reward"""
        self.reward_history.append(is_correct)

    @property
    def reward(self):
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)

    @property
    def visits(self):
        return len(self.reward_history)

    @property
    def corrects(self):
        return sum(self.reward_history)

    @property
    def variance(self):
        n = len(self.reward_history)
        if n <= 1:
            return 0.0
        mean = self.reward
        return sum((r - mean) ** 2 for r in self.reward_history) / (n - 1)

    def compute_informative_score(self, alpha=0.5, beta=0.3, gamma=0.2):
        """基于难度、reward 增益、波动度的综合信息度指标"""
        difficulty = 1 - (self.baseline_reward or 0.0)
        reward_gain = self.reward_history[-1] - (self.baseline_reward or 0.0)
        self.informative_score = (
            alpha * difficulty + beta * reward_gain + gamma * self.variance
        )


class DynamicSamplePool:
    @abstractmethod
    def _evict_oldest(self):
        pass
    @abstractmethod
    def add_or_update(self, sample: PoolSample, is_correct):
        pass
    @abstractmethod
    def sample(self, type: SampleType, k=5, temp=0.1):
        pass
    @abstractmethod
    def compute_cpool(self):
        pass
    @abstractmethod
    def initialize(self, dataset, evaluator, current_prompt: str):
        pass