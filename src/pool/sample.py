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

    def compute_informative_score(
            self,
            weights,           # Tensor / list, shape = [3]
            window:int = 3
            ):
        """
        “这个样本对当前策略 / 提示 / 控制器是否仍然具有学习价值”
        信息度指标:
        - difficulty: 基于 baseline 的难度 （初始太难或太简单的噪声的可能性很高， 高成本 + 低可学习信号）
        - reward_gain: 最近窗口内的平均增益 (比单次更稳健，将来可作为学习内容的潜力)
        - variance: reward 的波动度，结构敏感
        自动归一化三项后取平均，避免人工调参
        """
        if not self.reward_history:
            self.informative_score = 0.0
            return self.informative_score

        # 1. difficulty (难度)
        difficulty = (self.baseline_reward or 0.0) * (1 - (self.baseline_reward or 0.0))

        # 2. reward_gain (最近窗口平均提升)
        recent = self.reward_history[-window:] if len(self.reward_history) >= window else self.reward_history
        avg_recent = sum(recent) / len(recent)
        # 剩余学习空间（将来可作为学习内容的潜力）
        reward_gain = max(0.0, 1.0 - avg_recent)

        # 3. variance (波动)
        var = self.variance

        # ---- 自动归一化到 [0,1] ----
        def normalize(x, max_val=1.0):
            return min(max(x / (max_val + 1e-6), 0.0), 1.0)

        diff_n = normalize(difficulty)
        gain_n = normalize(reward_gain)
        var_n  = normalize(var)

        # 综合信息度 = 原始三者平均，现在自动学习权重
        self.informative_score = (weights[0] * diff_n 
                                  + weights[1] * gain_n 
                                  + weights[2] * var_n)
        return self.informative_score
