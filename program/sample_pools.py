import random
from collections import deque
from logger import logger

class PoolSample:
    def __init__(self, raw_sample:dict):
        self.raw = raw_sample  # 原始 dict, 包含 question/answer/choices
        self.visits = 0
        self.corrects = 0
        self.reward = 0.0

    def update(self, is_correct):
        self.visits += 1
        self.corrects += int(is_correct)
        self.reward = self.corrects / self.visits

class DynamicSamplePool:
    def __init__(self, max_size=1000, low=0.5, high=0.9):
        self.max_size = max_size
        self.low_threshold = low
        self.high_threshold = high
        self.hard = []
        self.mix = []
        self.success = []
        self.order = deque()  # 按加入顺序维护

    def _evict_oldest(self):
        if not self.order:
            return
        old = self.order.popleft()
        self.hard = [s for s in self.hard if s != old]
        self.mix = [s for s in self.mix if s != old]
        self.success = [s for s in self.success if s != old]

    def add_or_update(self, sample: PoolSample, is_correct):
        # 更新统计
        sample.update(is_correct)

        # 移除旧池
        self.hard = [s for s in self.hard if s != sample]
        self.mix = [s for s in self.mix if s != sample]
        self.success = [s for s in self.success if s != sample]

        # 根据 reward 放入新池
        r = sample.reward
        if r < self.low_threshold:
            self.hard.append(sample)
        elif r >= self.high_threshold:
            self.success.append(sample)
        else:
            self.mix.append(sample)

        # 容量控制
        self.order.append(sample)
        while len(self.order) > self.max_size:
            self._evict_oldest()

    def sample(self, pool="mixed", k=5):
        if pool == "hard":
            values = self.hard
        elif pool == "success":
            values = self.success
        elif pool == "mixed":
            values = self.hard + self.mix + self.success
        else:
            raise ValueError(f"Unknown pool: {pool}")
        return random.sample(values, min(k, len(values)))

    def initialize(self, dataset, evaluator, current_prompt: str, eval_repeat: int = 5):
        """dataset: list of raw sample dicts"""
        for raw in dataset:
            s = PoolSample(raw)

            for raw in dataset:
                # 把同一个样本复制 N 次
                repeated = [raw] * eval_repeat

                # 一次并行计算
                rewards = evaluator.batch_reward(current_prompt, repeated)

                # 统计
                s = PoolSample(raw)
                s.visits = eval_repeat
                s.corrects = int(rewards * eval_repeat)
                s.reward = rewards

                # 分类
                if s.reward < self.low_threshold:
                    self.hard.append(s)
                elif s.reward >= self.high_threshold:
                    self.success.append(s)
                else:
                    self.mix.append(s)

                # 顺序队列
                self.order.append(s)
                if len(self.order) > self.max_size:
                    self._evict_oldest()