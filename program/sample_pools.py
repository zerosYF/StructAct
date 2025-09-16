import random
from collections import deque
from logger import logger
from program.sample_pool import PoolSample, DynamicSamplePool, SampleType
import math
import numpy as np
    
class ContinuousSamplePool(DynamicSamplePool):
    """
    样本池：存储任务样本，并根据正/负反馈动作需求进行采样。
    - Negative 动作：从失败/不稳定样本中抽取 -> 用于反思改写
    - Positive 动作：从成功/稳定样本中抽取 -> 用于归纳改写
    """
    def __init__(self, max_size=1000, 
                 high_reward_threshold=0.9,
                 low_reward_threshold=0.3,
                 var_unstable=0.05,
                 informative_threshold=0.5,

                 negative_informative_mag=0.2,
                 negative_var_mag=0.5,
                 positive_informative_mag=0.5,
                 positive_var_mag=0.2,
                 ):
        self.max_size = max_size
        self.samples:list[PoolSample] = []
        self.order = deque()

        self.high_reward_threshold = high_reward_threshold
        self.low_reward_threshold = low_reward_threshold
        self.var_unstable = var_unstable
        self.informative_threshold = informative_threshold

        self.negative_informative_mag = negative_informative_mag
        self.negative_var_mag = negative_var_mag

        self.positive_informative_mag = positive_informative_mag
        self.positive_var_mag = positive_var_mag

    def _evict_oldest(self):
        '''
        样本池容量控制维护 
        '''
        if not self.order:
            return
        old = self.order.popleft()
        self.samples = [s for s in self.samples if s != old]

    def add_or_update(self, sample: PoolSample, is_correct):
        sample.update(is_correct)
        sample.compute_informative_score()
        if sample not in self.samples:
            self.samples.append(sample)
        self.order.append(sample)
        while len(self.order) > self.max_size:
            self._evict_oldest()
        logger.info(
            f"Sample updated: visits={sample.visits}, reward={sample.reward:.3f}, "
            f"informative_score={sample.informative_score:.3f}"
        )

    def _softmax(vals: np.ndarray, temperature: float = 1.0):
        vals = vals - np.max(vals)  # 防止 exp 溢出
        exp_vals = np.exp(vals / max(temperature, 1e-6))
        probs = exp_vals / np.sum(exp_vals)
        return probs

    def sample(self, type: SampleType, k=5, temperature=1.0):
        """根据动作抽样样本"""
        if not self.samples:
            return []

        scores = []
        for s in self.samples:
            if type == SampleType.Negative:
                sc = self.priority_for_negative(s)
            elif type == SampleType.Positive:
                sc = self.priority_for_positive(s)
            else:
                raise ValueError(f"Unknown action {type}")
            scores.append((s, sc))

        vals = np.array([sc for _, sc in scores])
        probs = self._softmax(vals, temperature)
        selected = np.random.choice(
            [s for s, _ in scores], size=min(k, len(scores)), replace=False, p=probs
        )
        return list(selected)
    
    # -------- 动作优先级函数 --------
    def priority_for_negative(self, sample: PoolSample):
        """
        负反馈动作优先级：
        - 奖励越低，越值得反思
        - variance 越大（不稳定），越值得关注
        - informative 高，说明样本对任务有启发性
        """
        var_norm = min(1.0, sample.variance / (self.var_unstable + 1e-6))
        return ((1 - sample.reward) 
            + self.negative_informative_mag * sample.informative_score 
            +  self.negative_var_mag * var_norm)

    def priority_for_positive(self, sample: PoolSample):
        """
        正反馈动作优先级：
        - reward 高于 baseline，说明有改进
        - informative 高，说明样本对任务有启发性
        - variance 小，说明表现稳定
        """
        var_penalty = min(1.0, sample.variance / (self.var_unstable + 1e-6))
        reward_gain = sample.reward - (sample.baseline_reward or 0.0)
        return (reward_gain 
            + self.positive_informative_mag * sample.informative_score
            - self.positive_var_mag * var_penalty)
    
    def compute_cpool(self):
        """
        以阈值分池判断
        """

        total = len(self.samples)
        if total == 0:
            return 0.0

        # counters / aggregated scores
        easy_score = 0.0            # sum of confidence-weighted easy-success
        informative_score = 0.0    # sum of confidence-weighted informative-success
        hard_score = 0.0           # sum of hard indicators

        for s in self.samples:
            # confidence for this sample's success probability
            conf_success = self._wilson_lower_bound(s.corrects, s.visits) if s.visits > 0 else 0.0
            # classify by thresholds
            is_high = s.reward >= self.high_reward_threshold
            is_unstable = s.variance > self.var_unstable

            # easy-success: high reward, low variance, baseline also high
            # 样本池正向指标
            if is_high and not is_unstable:
                easy_score += conf_success

            # informative-success: high reward, informative_score high (baseline low often)
            informative_flag = (s.informative_score or 0.0) >= self.informative_threshold
            if is_high and informative_flag:
                informative_score += conf_success

            # hard: low reward
            is_low = s.reward < self.low_reward_threshold
            if is_low:
                hard_score += 1.0

        # normalize to ratios in [0,1] style (divide by total)
        easy_ratio = easy_score / total
        informative_ratio = informative_score / total
        hard_ratio = hard_score / total

        # linear combination
        cpool_raw = (self.cpool_weights["easy"] * easy_ratio +
                    self.cpool_weights["informative"] * informative_ratio  -
                    self.cpool_weights["hard"] * hard_ratio)

        # optional normalization: map to [0,1] by sigmoid or clamping
        # here use a soft clamp: sigmoid-like scaling
        cpool = 1 / (1 + math.exp(-cpool_raw))  # sigmoid -> in (0,1)
        # if you prefer linear clamp:
        # cpool = max(0.0, min(1.0, cpool_raw))

        # also return diagnostics if you want:
        diagnostics = {
            "easy_ratio": easy_ratio,
            "informative_ratio": informative_ratio,
            "hard_ratio": hard_ratio,
            "cpool_raw": cpool_raw,
            "cpool": cpool,
            "total_samples": total
        }
        logger.info(f"compute_cpool: {diagnostics}")
        return diagnostics
    
    def _wilson_lower_bound(self, successes, n, z=1.96):
        """
            把访问少但表面成功的样本打低分，避免噪声样本把 easy_ratio / informative_ratio 抬高。
        """
        if n == 0:
            return 0.0
        phat = successes / n
        z2 = z * z
        denom = 1 + z2 / n
        centre = phat + z2 / (2 * n)
        root = math.sqrt(max(0.0, phat * (1 - phat) / n + z2 / (4 * n * n)))
        lower = (centre - z * root) / denom
        return max(0.0, lower)

    def initialize(self, dataset, evaluator, current_prompt: str):
            """批量初始化"""
            logger.info(f"Initializing pool with {len(dataset)} samples...")
            rewards = evaluator.batch_reward_n(current_prompt, dataset)
            for raw, r in zip(dataset, rewards):
                s = PoolSample(raw)
                s.update(r)
                s.baseline_reward = r
                s.compute_informative_score()
                self.samples.append(s)
                self.order.append(s)
                if len(self.order) > self.max_size:
                    self._evict_oldest()
            logger.info(f"Pool initialized with {len(self.samples)} samples.")