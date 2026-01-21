from src.logger import logger
from pool.sample import PoolSample, SampleType
from src.net.controller import SurrogateNetController
import math
import numpy as np
import torch
import torch.nn.functional as F
from config import SearchConfig
    
class DynamicSamplePool:
    """
    Sample Pool = external belief state
    - search only reads
    - environment feedback updates belief slowly
    """
    def __init__(self,  
                 config:SearchConfig,
                 high_reward_threshold=0.9,
                 low_reward_threshold=0.3,
                 var_unstable=0.05,
                 informative_threshold=0.5,
                 pool_update_interval=10,
                 ):
        self.config = config
        self.samples:list[PoolSample] = []
        self.pending_updates:list[tuple[PoolSample, float]] = []

        self.pool_step = 0
        self.pool_update_interval = pool_update_interval

        self.net_controller = SurrogateNetController(
            lr=self.config.net_lr,
            device=self.config.net_device,
            buffer_size=self.config.net_buffer_size
        )

        # 分位法控制
        self.high_reward_threshold = high_reward_threshold
        self.low_reward_threshold = low_reward_threshold
        self.var_unstable = var_unstable  # variance归一化尺度
        self.informative_threshold = informative_threshold
    
    def observe(self, sample:PoolSample, reward:float):
        self.pending_updates.append((sample, reward))
    
    def step(self):
        self.pool_step += 1
        if self.pool_step % self.pool_update_interval == 0:
            self._flush_update()
    
    def _flush_update(self):
        if not self.pending_updates:
            return
        logger.info(f"Flushing {len(self.pending_updates)} pending updates to sample pool...")
        informative_weights = self.net_controller.bundle.get_informative_weights().tolist()

        for sample, reward in self.pending_updates:
            sample.update(reward)
            sample.compute_informative_score(informative_weights)
            if sample not in self.samples:
                self.samples.append(sample)
        self.pending_updates.clear()

    def sample(self, type: SampleType, k=5, temperature=1.0):
        """根据动作抽样样本"""
        if not self.samples:
            return []

        scores = []
        weights = self.net_controller.bundle.get_pool_weights().tolist()
        for s in self.samples:
            if type == SampleType.Negative:
                sc = self.priority_for_negative(s, weights)
            elif type == SampleType.Positive:
                sc = self.priority_for_positive(s, weights)
            else:
                raise ValueError(f"Unknown action {type}")
            scores.append((s, sc))

        vals = np.array([sc for _, sc in scores])
        probs = F.softmax(torch.tensor(vals)/ max(temperature, 1e-6), dim=0).numpy()
        selected = np.random.choice(
            [s for s, _ in scores], size=min(k, len(scores)), replace=False, p=probs
        )
        return list(selected)
    
    # -------- 动作优先级函数 --------
    def priority_for_negative(self, sample: PoolSample, weights:list):
        """
        负反馈动作优先级：
        - 奖励越低，越值得反思
        - variance 越大（不稳定），越值得关注
        - informative 高，说明样本对任务有启发性
        """
        var_norm = min(1.0, sample.variance / (self.var_unstable + 1e-6))
        reward_term = 1.0 - sample.mean
        var_term = +var_norm

        return (
            weights[0] * reward_term +
            weights[1] * sample.informative_score +
            weights[2] * var_term
        )

    def priority_for_positive(self, sample: PoolSample, weights:list):
        """
        正反馈动作优先级：
        - reward 高于 baseline，说明有改进
        - informative 高，说明样本对任务有启发性
        - variance 小，说明表现稳定
        """
        var_penalty = min(1.0, sample.variance / (self.var_unstable + 1e-6))
        reward_term = sample.mean - (sample.base_mean or 0.0)
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
        total = len(self.samples)
        if total == 0:
            return 0.0

        # counters / aggregated scores
        easy_score = 0.0
        informative_score = 0.0
        hard_score = 0.0


        rewards = [s.mean for s in self.samples]
        if rewards:
            q_high = np.percentile(rewards, int(self.high_reward_threshold * 100))
            q_low = np.percentile(rewards, int(self.low_reward_threshold * 100))
        else:
            q_high, q_low = 0.0, 0.0, 0.0


        for s in self.samples:
            # confidence for this sample's success probability
            conf_success = s.wilson_lower_bound()

            # -------- 1: 相对提升 --------
            improvement = s.mean - s.base_mean

            # -------- 2: 分位数判断 --------
            is_high = s.mean >= q_high or improvement > 0
            is_low = s.mean <= q_low and improvement <= 0

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
    
    
    def get_net_controller(self):
        return self.net_controller

    def initialize(self, dataset, evaluator, current_prompt: str):
            """批量初始化"""
            logger.info(f"Initializing pool with {len(dataset)} samples...")
            rewards = evaluator.batch_reward_n(current_prompt, dataset)
            init_informative_weights = self.net_controller.bundle.get_informative_weights().tolist()
            for raw, r in zip(dataset, rewards):
                s = PoolSample(raw)
                s.update(r)
                s.freeze_baseline()
                s.compute_informative_score(init_informative_weights)
                self.samples.append(s)
            logger.info(f"Pool initialized with {len(self.samples)} samples.")