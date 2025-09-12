import random
from collections import deque
from logger import logger
from program.sample_pool import PoolSample, DynamicSamplePool, SampleType
import math
import numpy as np

class BucketedSamplePool(DynamicSamplePool):
    def __init__(self, max_size=1000, low=0.3, high=0.9):
        self.max_size = max_size
        self.low_threshold = low
        self.high_threshold = high
        self.hard:list[PoolSample] = []
        self.mix:list[PoolSample] = []
        self.success:list[PoolSample] = []
        self.informative_success:list[PoolSample] = []
        self.order = deque() 

        self.hard_ratio = 0.7  # hard样本占比
        self.success_ratio = 0.7  # success样本占比

        self.cpool_lambda = 0.5
        self.cpool_mu = 0.3

    def _evict_oldest(self):
        if not self.order:
            return
        old = self.order.popleft()
        self.hard = [s for s in self.hard if s != old]
        self.mix = [s for s in self.mix if s != old]
        self.success = [s for s in self.success if s != old]
        self.informative_success = [s for s in self.informative_success if s != old]

    def add_or_update(self, sample: PoolSample, is_correct):
        sample.update(is_correct)
        logger.info(f"Sample updated: visits={sample.visits}, corrects={sample.corrects}, reward={sample.reward:.3f}")    
        
        self.hard = [s for s in self.hard if s != sample]
        self.mix = [s for s in self.mix if s != sample]
        self.success = [s for s in self.success if s != sample]
        self.informative_success = [s for s in self.informative_success if s != sample]

        sample.compute_informative_score()

        r = sample.reward
        if r < self.low_threshold:
            self.hard.append(sample)
        elif r >= self.high_threshold:
            if sample.informative_score > 0.5:
                self.informative_success.append(sample)
            else:
                self.success.append(sample)
        else:
            self.mix.append(sample)

        self.order.append(sample)
        while len(self.order) > self.max_size:
            self._evict_oldest()
        
        logger.info(f"Pool sizes: hard={len(self.hard)}, mix={len(self.mix)}, success={len(self.success)}, informative_success={len(self.informative_success)}")
    
    def _uct_score(self, sample: PoolSample, total_visits: int, pool_type:str, exploration_weight=1.0):
        exploration_term = exploration_weight * math.sqrt(math.log(total_visits + 1) / (sample.visits + 1))
        if pool_type == "hard":
            score = - sample.reward + exploration_term
        elif pool_type in ("success", "informative_success"):
            score = sample.reward + exploration_term
        else:  
            score = sample.reward
        return score

    def sample(self, type:SampleType, k=5):
        """
        pool: "positive" or "negative"
        k: total number of samples
        success_ratio: positive时从 success/informative_success 中采样比例
        hard_ratio: negative时从 hard 中采样比例
        """
        selected = []

        if type == SampleType.Positive:
            num_success = int(k * self.success_ratio)
            num_mix = k - num_success

            # success/informative_success
            success_pool = self.informative_success if self.informative_success else self.success
            if success_pool and num_success > 0:
                scored_samples = [
                    (s, self._uct_score(s, sum(smp.visits for smp in success_pool), "success", 0.8))
                    for s in success_pool
                ]
                scored_samples.sort(key=lambda x: x[1], reverse=True)
                selected += [s for s, _ in scored_samples[:num_success]]

            # mix 补齐
            if self.mix and num_mix > 0:
                weights = [1 - abs(s.reward - 0.5) for s in self.mix]
                total = sum(weights)
                probs = [w / total for w in weights]
                selected += random.choices(self.mix, weights=probs, k=min(num_mix, len(self.mix)))

        elif type == SampleType.Negative:
            num_hard = int(k * self.hard_ratio)
            num_mix = k - num_hard

            # hard
            if self.hard and num_hard > 0:
                scored_samples = [
                    (s, self._uct_score(s, sum(smp.visits for smp in self.hard), "hard", 1.5))
                    for s in self.hard
                ]
                scored_samples.sort(key=lambda x: x[1], reverse=True)
                selected += [s for s, _ in scored_samples[:num_hard]]

            # mix 补齐
            if self.mix and num_mix > 0:
                weights = [1 - abs(s.reward - 0.5) for s in self.mix]
                total = sum(weights)
                probs = [w / total for w in weights]
                selected += random.choices(self.mix, weights=probs, k=min(num_mix, len(self.mix)))

        else:
            raise ValueError(f"Unknown pool type: {type}")

        logger.info(f"Sampled {len(selected)} samples for pool='{type}' (k={k})")
        return selected
    
    def compute_cpool(self):
        """
        Compute cpool and return full diagnostics instead of only a float.
        """
        total = len(self.hard) + len(self.mix) + len(self.success) + len(self.informative_success)
        if total == 0:
            diagnostics = {
                "cpool": 0.0,
                "success_ratio": 0.0,
                "hard_ratio": 0.0,
                "mix_ratio": 0.0,
                "total_samples": 0,
            }
            return diagnostics

        hard_ratio = len(self.hard) / total
        mix_ratio = len(self.mix) / total
        success_ratio = (len(self.success) + len(self.informative_success)) / total

        cpool = success_ratio - self.cpool_lambda * hard_ratio - self.cpool_mu * mix_ratio

        diagnostics = {
            "cpool": cpool,
            "success_ratio": success_ratio,
            "hard_ratio": hard_ratio,
            "mix_ratio": mix_ratio,
            "total_samples": total,
        }

        logger.info(
            f"C_pool computed: {cpool:.3f} "
            f"(success={success_ratio:.2f}, hard={hard_ratio:.2f}, mix={mix_ratio:.2f})"
        )

        return diagnostics

    def initialize(self, dataset, evaluator, current_prompt: str):
        """dataset: list of raw sample dicts"""
        logger.info(f"Initializing sample pool with {len(dataset)} samples...")

        rewards = evaluator.batch_reward_n(current_prompt, dataset)  # list[float], 长度 = len(dataset)

        for raw, r in zip(dataset, rewards):
            s = PoolSample(raw)
            s.update(r)
            s.baseline_reward = r
            s.compute_informative_score()

            if s.reward < self.low_threshold:
                self.hard.append(s)
            elif s.reward >= self.high_threshold:
                if s.informative_score > 0.5:
                    self.informative_success.append(s)
                else:
                    self.success.append(s)
            else:
                self.mix.append(s)

            self.order.append(s)
            if len(self.order) > self.max_size:
                self._evict_oldest()

        logger.info(
            f"Sample pool initialized: "
            f"hard={len(self.hard)}, mix={len(self.mix)}, success={len(self.success)}, informative_success={len(self.informative_success)}"
        )
    
class ContinuousSamplePool(DynamicSamplePool):
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.samples:list[PoolSample] = []
        self.order = deque()

        self.cpool_high_threshold = 0.9
        self.cpool_low_threshold = 0.3
        self.cpool_var_unstable = 0.05
        self.cpool_baseline_high = 0.8
        self.cpool_informative_thresh = 0.5
        self.cpool_weights = {
                "easy": 1.0,
                "informative": 1.5,
                "mix": 0.8,
                "hard": 1.0
            }

    def _evict_oldest(self):
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

    def sample(self, type: SampleType, k=5, temp=0.1):
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

        scores = [(s, sc) for s, sc in scores if sc > 0]
        if not scores:
            return []

        vals = np.array([sc for _, sc in scores])
        probs = np.exp(vals / temp) / np.sum(np.exp(vals / temp))
        selected = np.random.choice(
            [s for s, _ in scores], size=min(k, len(scores)), replace=False, p=probs
        )
        return list(selected)
    
    # -------- 动作优先级函数 --------
    def priority_for_negative(self, sample: PoolSample, w1=0.7, w2=0.3):
        """负反馈：奖励低、variance 高的样本更优先"""
        return w1 * (1 - sample.reward) + w2 * sample.variance

    def priority_for_positive(self, sample: PoolSample, a1=0.6, a2=0.3, a3=0.1):
        """正反馈：奖励高、baseline 低且 variance 小的样本更优先"""
        reward_gain = sample.reward - sample.baseline_reward
        nontrivial = 1.0 - sample.baseline_reward
        stable = 1.0 if sample.variance < 0.05 else 0.0
        return a1 * reward_gain + a2 * sample.reward * nontrivial + a3 * stable
    
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
        mix_score = 0.0            # sum of mix indicators
        hard_score = 0.0           # sum of hard indicators

        for s in self.samples:
            # confidence for this sample's success probability
            conf_success = self._wilson_lower_bound(s.corrects, s.visits) if s.visits > 0 else 0.0
            # classify by thresholds
            is_high = s.reward >= self.cpool_high_threshold
            is_low = s.reward < self.cpool_low_threshold
            is_unstable = s.variance > self.cpool_var_unstable
            baseline_high_flag = (s.baseline_reward or 0.0) >= self.cpool_baseline_high
            informative_flag = (s.informative_score or 0.0) >= self.cpool_informative_thresh

            # easy-success: high reward, low variance, baseline also high
            if is_high and not is_unstable and baseline_high_flag:
                easy_score += conf_success

            # informative-success: high reward, informative_score high (baseline low often)
            if is_high and informative_flag:
                informative_score += conf_success

            # mix: middle reward or high variance (uncertain)
            if (not is_high) and (not is_low):
                # reward in (low, high) -> mix
                mix_score += 1.0
            if is_unstable:
                # treat unstable as contributing to mix (has uncertainty)
                mix_score += 0.5

            # hard: low reward
            if is_low:
                hard_score += 1.0

        # normalize to ratios in [0,1] style (divide by total)
        easy_ratio = easy_score / total
        informative_ratio = informative_score / total
        mix_ratio = mix_score / total
        hard_ratio = hard_score / total

        # linear combination
        cpool_raw = (self.cpool_weights["easy"] * easy_ratio +
                    self.cpool_weights["informative"] * informative_ratio -
                    self.cpool_weights["mix"] * mix_ratio -
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
            "mix_ratio": mix_ratio,
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