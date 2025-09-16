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

    def compute_informative_score(self, window:int = 3):
        """
        信息度指标:
        - difficulty: 基于 baseline 的难度
        - reward_gain: 最近窗口内的平均增益 (比单次更稳健)
        - variance: reward 的波动度
        自动归一化三项后取平均，避免人工调参
        """
        if not self.reward_history:
            self.informative_score = 0.0
            return self.informative_score

        # 1. difficulty (难度)
        difficulty = 1 - (self.baseline_reward or 0.0)

        # 2. reward_gain (最近窗口平均提升)
        recent = self.reward_history[-window:] if len(self.reward_history) >= window else self.reward_history
        avg_recent = sum(recent) / len(recent)
        reward_gain = avg_recent - (self.baseline_reward or 0.0)

        # 3. variance (波动)
        var = self.variance

        # ---- 自动归一化到 [0,1] ----
        def normalize(x, max_val=1.0):
            return min(max(x / (max_val + 1e-6), 0.0), 1.0)

        diff_n = normalize(difficulty)
        gain_n = normalize(reward_gain)
        var_n  = normalize(var)

        # 综合信息度 = 三者平均
        self.informative_score = (diff_n + gain_n + var_n) / 3.0
        return self.informative_score


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