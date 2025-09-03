import random
class DynamicSamplePool:
    def __init__(self, max_size=1000):
        self.hard = {}
        self.success = {}
        self.history = {}   # input_id -> list of (prompt_id, reward)

    def _make_key(self, input_text, gold):
        return hash((input_text, gold))  # 样本唯一标识

    def add(self, input_text, output, gold, reward, prompt_id):
        key = self._make_key(input_text, gold)
        record = {"input": input_text, "output": output,
                  "gold": gold, "reward": reward, "prompt_id": prompt_id}

        # 更新历史
        if key not in self.history:
            self.history[key] = []
        self.history[key].append((prompt_id, reward))

        # 放入池（注意迁移）
        if reward == 1:
            self.success[key] = record
            if key in self.hard:
                del self.hard[key]   # 从 hard 迁移到 success
        else:
            self.hard[key] = record
            if key in self.success:
                del self.success[key]   # 从 success 迁移回 hard

    def sample(self, pool="mixed", k=4):
        if pool == "hard":
            values = list(self.hard.values())
        elif pool == "success":
            values = list(self.success.values())
        elif pool == "mixed":
            values = list(self.hard.values()) + list(self.success.values())
        else:
            raise ValueError(f"Unknown pool: {pool}")

        return random.sample(values, min(k, len(values)))