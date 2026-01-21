from enum import Enum
import math
class SampleType(Enum):
    Positive = "positive"
    Negative = "negative"

#Beta-Binomial
class PoolSample:
    def __init__(self, raw_sample: dict):
        self.raw = raw_sample  # question/answer/choices
        # posterior
        self.alpha = 1.0
        self.beta  = 1.0
        # baseline posterior (frozen)
        self.base_alpha = 1.0
        self.base_beta  = 1.0
        self.informative_score = 0.0

    def update(self, reward: float):
        # reward ∈ [0,1]
        self.alpha += reward
        self.beta  += (1 - reward)
    
    def freeze_baseline(self):
        self.base_alpha = self.alpha
        self.base_beta  = self.beta

    @property
    def visits(self):
        return self.alpha + self.beta - 2  # initial 1,1 not counted
    
    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def base_mean(self):
        return self.base_alpha / (self.base_alpha + self.base_beta)
    
    @property
    def uncertainty(self):
        return 1.0 / (self.alpha + self.beta + 1)

    @property
    def variance(self):
        a, b = self.alpha, self.beta
        denom = (a + b) ** 2 * (a + b + 1)
        return (a * b) / denom if denom > 0 else 0.0
    
    def wilson_lower_bound(self, z=1.96):
        """
        Wilson lower bound approximation
        把访问少但表面成功的样本打低分，避免噪声样本把 easy_ratio / informative_ratio 抬高。
        在小样本阶段抑制对全局判断的过度影响。
        """
        n = self.visits
        if n <= 0:
            return 0.0

        phat = (self.alpha - 1) / n
        z2 = z * z
        denom = 1 + z2 / n
        centre = phat + z2 / (2 * n)
        root = math.sqrt(
            max(0.0, phat * (1 - phat) / n + z2 / (4 * n * n))
        )
        lower = (centre - z * root) / denom
        return max(0.0, lower)

    def compute_informative_score(self, weights):
        """
        “这个样本对当前策略 / 提示 / 控制器是否仍然具有学习价值”
        信息度指标:
        - difficulty: 基于 baseline 的难度 （初始太难或太简单的噪声的可能性很高， 高成本 + 低可学习信号）
        - reward_gain: 最近平均增益 (比单次更稳健，将来可作为学习内容的潜力)
        - variance: reward 的波动度，结构敏感
        自动归一化三项后取平均，避免人工调参
        """
        mean = self.mean
        var = self.variance

        # 1. difficulty (难度)
        diff_n = mean * (1.0 - mean) * 4.0  # [0,1]，在 0.5 最大

        # 2. reward_gain (最近平均提升)
        gain = max(0.0, mean - self.base_mean)
        gain_n = gain  # already in [0,1]

        # 3. variance (波动)
        var_n = min(1.0, var * 12.0)

        # 综合信息度 = 原始三者平均，现在自动学习权重
        self.informative_score = (weights[0] * diff_n 
                                  + weights[1] * gain_n 
                                  + weights[2] * var_n)
        return self.informative_score
