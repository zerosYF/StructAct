import random
from collections import deque
from logger import logger
import math

class PoolSample:
    def __init__(self, raw_sample:dict):
        self.raw = raw_sample  # question/answer/choices
        self.visits = 0
        self.corrects = 0
        self.reward = 0.0

    def update(self, is_correct):
        self.visits += 1
        self.corrects += int(is_correct)
        self.reward = self.corrects / self.visits

class DynamicSamplePool:
    def __init__(self, max_size=1000, low=0.3, high=0.9):
        self.max_size = max_size
        self.low_threshold = low
        self.high_threshold = high
        self.hard = []
        self.mix = []
        self.success = []
        self.order = deque() 

    def _evict_oldest(self):
        if not self.order:
            return
        old = self.order.popleft()
        self.hard = [s for s in self.hard if s != old]
        self.mix = [s for s in self.mix if s != old]
        self.success = [s for s in self.success if s != old]

    def add_or_update(self, sample: PoolSample, is_correct):
        sample.update(is_correct)
        logger.info(f"Sample updated: visits={sample.visits}, corrects={sample.corrects}, reward={sample.reward:.3f}")    
        
        self.hard = [s for s in self.hard if s != sample]
        self.mix = [s for s in self.mix if s != sample]
        self.success = [s for s in self.success if s != sample]

        r = sample.reward
        if r < self.low_threshold:
            self.hard.append(sample)
        elif r >= self.high_threshold:
            self.success.append(sample)
        else:
            self.mix.append(sample)

        self.order.append(sample)
        while len(self.order) > self.max_size:
            self._evict_oldest()
        
        logger.info(f"Pool sizes: hard={len(self.hard)}, mix={len(self.mix)}, success={len(self.success)}")
    
    def _uct_score(self, sample: PoolSample, total_visits: int, pool_type:str, exploration_weight=1.0):
        exploration_term = exploration_weight * math.sqrt(math.log(total_visits + 1) / (sample.visits + 1))
        if pool_type == "hard":
            score = - sample.reward + exploration_term
        elif pool_type == "success":
            score = sample.reward + exploration_term
        else:  
            score = sample.reward
        return score

    def sample(self, pool="mixed", k=5):
        if pool == "hard":
            values = self.hard
        elif pool == "success":
            values = self.success
        elif pool == "mixed":
            values = self.mix
        else:
            raise ValueError(f"Unknown pool: {pool}")
        logger.info(f"Sampling {k} from pool '{pool}' with {len(values)} available samples.")
        if pool == "mixed":
            selected = random.sample(values, min(k, len(values)))
            logger.info(f"Selected {len(selected)} samples from mixed pool.")
        else:
            scored_samples = [(s, self._uct_score(s, sum(smp.visits for smp in values), pool)) for s in values]
            scored_samples.sort(key=lambda x: x[1], reverse=True)
            selected = [s for s, score in scored_samples[:k]]
            logger.info(f"Selected {len(selected)} samples from {pool} pool based on UCT scores.")
        return selected
    
    def compute_cpool(self, lambda_=0.5, mu=0.3):
        total = len(self.hard) + len(self.mix) + len(self.success)
        if total == 0:
            return 0.0

        hard_ratio = len(self.hard) / total
        mix_ratio = len(self.mix) / total
        success_ratio = len(self.success) / total

        cpool = success_ratio - lambda_ * hard_ratio - mu * mix_ratio
        logger.info(
            f"C_pool computed: {cpool:.3f} "
            f"(success={success_ratio:.2f}, hard={hard_ratio:.2f}, mix={mix_ratio:.2f})"
        )
        return cpool

    def initialize(self, dataset, evaluator, current_prompt: str):
        """dataset: list of raw sample dicts"""
        logger.info(f"Initializing sample pool with {len(dataset)} samples...")

        rewards = evaluator.batch_reward_n(current_prompt, dataset)  # list[float], 长度 = len(dataset)

        for raw, r in zip(dataset, rewards):
            s = PoolSample(raw)
            s.visits = 1
            s.corrects = int(r)   
            s.reward = float(r)  

            if s.reward < self.low_threshold:
                self.hard.append(s)
            elif s.reward >= self.high_threshold:
                self.success.append(s)
            else:
                self.mix.append(s)

            self.order.append(s)
            if len(self.order) > self.max_size:
                self._evict_oldest()

        logger.info(
            f"Sample pool initialized: "
            f"hard={len(self.hard)}, mix={len(self.mix)}, success={len(self.success)}"
        )