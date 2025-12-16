import random
from collections import deque
from logger import logger
from program.sample_pool import PoolSample, DynamicSamplePool, SampleType
from micronet.parameters import ParamBundle
import math
import numpy as np
    
class ContinuousSamplePool(DynamicSamplePool):
    """
    样本池：存储任务样本，并根据正/负反馈动作需求进行采样。
    - Negative 动作：从失败/不稳定样本中抽取 -> 用于反思改写
    - Positive 动作：从成功/稳定样本中抽取 -> 用于归纳改写
    """
    def __init__(self, 
                 param_bundle: ParamBundle,
                 max_size=1000, 
                 high_reward_threshold=0.9,
                 low_reward_threshold=0.3,
                 var_unstable=0.05,
                 informative_threshold=0.5,
                 ):
        self.param_bundle = param_bundle
        self.max_size = max_size
        self.samples:list[PoolSample] = []
        self.order = deque()

        # 分位法控制
        self.high_reward_threshold = high_reward_threshold
        self.low_reward_threshold = low_reward_threshold
        
        self.var_unstable = var_unstable  # variance归一化尺度
        self.informative_threshold = informative_threshold

    def _evict_oldest(self):
        '''
        样本池容量控制维护 
        '''
        if not self.order:
            return
        old = self.order.popleft()
        self.samples = [s for s in self.samples if s != old]

    def add_or_update(self, sample: PoolSample, is_correct, informative_weights):
        sample.update(is_correct)
        sample.compute_informative_score(informative_weights)
        if sample not in self.samples:
            self.samples.append(sample)
        self.order.append(sample)
        while len(self.order) > self.max_size:
            self._evict_oldest()
        logger.info(
            f"Sample updated: visits={sample.visits}, reward={sample.reward:.3f}, "
            f"informative_score={sample.informative_score:.3f}"
        )


    def sample(self, type: SampleType, k=5, temperature=1.0):
        def _softmax(vals: np.ndarray, temperature:float):
            vals = vals - np.max(vals)  # 防止 exp 溢出
            exp_vals = np.exp(vals / max(temperature, 1e-6))
            probs = exp_vals / np.sum(exp_vals)
            return probs

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
        probs = _softmax(vals, temperature)
        selected = np.random.choice(
            [s for s, _ in scores], size=min(k, len(scores)), replace=False, p=probs
        )
        return list(selected)
    
    # -------- 动作优先级函数 --------
    def priority_for_negative(self, sample: PoolSample, weights):
        """
        负反馈动作优先级：
        - 奖励越低，越值得反思
        - variance 越大（不稳定），越值得关注
        - informative 高，说明样本对任务有启发性
        """
        var_norm = min(1.0, sample.variance / (self.var_unstable + 1e-6))
        reward_term = 1.0 - sample.reward
        var_term = +var_norm

        return (
            weights[0] * reward_term +
            weights[1] * sample.informative_score +
            weights[2] * var_term
        )

    def priority_for_positive(self, sample: PoolSample, weights):
        """
        正反馈动作优先级：
        - reward 高于 baseline，说明有改进
        - informative 高，说明样本对任务有启发性
        - variance 小，说明表现稳定
        """
        var_penalty = min(1.0, sample.variance / (self.var_unstable + 1e-6))
        reward_term = sample.reward - (sample.baseline_reward or 0.0)
        var_term = -var_penalty
        return (
            weights[0] * reward_term +
            weights[1] * sample.informative_score +
            weights[2] * var_term
        )
    
    def compute_cpool(self):
        """
        compute_cpool:
        - 不仅依赖绝对 reward 阈值
        - 增加相对提升 (baseline 对比)
        - 增加分位数判断 (避免低准确率阶段失效)
        """
        import numpy as np
        total = len(self.samples)
        if total == 0:
            return 0.0

        # counters / aggregated scores
        easy_score = 0.0
        informative_score = 0.0
        hard_score = 0.0


        rewards = [s.reward for s in self.samples]
        if rewards:
            q_high = np.percentile(rewards, int(self.high_reward_threshold * 100))
            q_low = np.percentile(rewards, int(self.low_reward_threshold * 100))
        else:
            q_high, q_low = 0.0, 0.0, 0.0


        for s in self.samples:
            # confidence for this sample's success probability
            conf_success = self._wilson_lower_bound(s.corrects, s.visits) if s.visits > 0 else 0.0

            # -------- 1: 相对提升 --------
            baseline_reward = getattr(s, "baseline_reward", 0.0)
            improvement = s.reward - baseline_reward

            # -------- 2: 分位数判断 --------
            is_high = s.reward >= q_high or improvement > 0
            is_low = s.reward <= q_low and improvement <= 0

            # -------- 3: 稳定性加权 --------
            is_unstable = s.variance > self.var_unstable

            # easy-success: 高 reward 或有提升，且稳定
            if is_high and not is_unstable:
                easy_score += conf_success

            # informative-success: 高 reward/有提升，且 informative 高
            informative_flag = (s.informative_score or 0.0) >= self.informative_threshold
            if is_high and informative_flag:
                informative_score += conf_success


            # hard: reward 低且无提升
            if is_low:
                hard_score += 1.0


        # normalize to ratios in [0,1]
        easy_ratio = easy_score / total
        informative_ratio = informative_score / total
        hard_ratio = hard_score / total
        cpool_raw = easy_ratio - (informative_ratio + hard_ratio)

        # sigmoid scaling
        cpool = 1 / (1 + math.exp(-cpool_raw))


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

    def initialize(self, dataset, evaluator, current_prompt: str, informative_weights):
            """批量初始化"""
            logger.info(f"Initializing pool with {len(dataset)} samples...")
            rewards = evaluator.batch_reward_n(current_prompt, dataset)
            for raw, r in zip(dataset, rewards):
                s = PoolSample(raw)
                s.update(r)
                s.baseline_reward = r
                s.compute_informative_score(informative_weights)
                self.samples.append(s)
                self.order.append(s)
                if len(self.order) > self.max_size:
                    self._evict_oldest()
            logger.info(f"Pool initialized with {len(self.samples)} samples.")