import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class LearnableParam:
    """用于区分可学习和不可学习参数。"""
    value: float
    learnable: bool = True
    min_val: float = 0.0
    max_val: float = 1.0

    def clamp(self):
        self.value = float(np.clip(self.value, self.min_val, self.max_val))


class ParamBundle:
    """
    一个统一管理超参数的类：
    - 负责存储参数
    - 向网络暴露参数向量
    - 从网络预测值更新参数
    - 写入 SamplePool / MCTS
    """

    def __init__(self):
        # ====== SamplePool 参数 ======
        self.sample_params = {
            "negative_informative_mag": LearnableParam(0.40, True, 0.0, 2.0),
            "positive_var_mag":         LearnableParam(0.95, True, 0.0, 2.0),
            "informative_threshold":    LearnableParam(0.10, True, 0.0, 1.0),
            "reward_temperature":       LearnableParam(0.70, True, 0.1, 5.0),
        }

        # ====== MCTS 参数 ======
        self.mcts_params = {
            "c_puct":     LearnableParam(1.5, True, 0.1, 5.0),
            "explore_eps": LearnableParam(0.1, True, 0.0, 1.0),
        }

        # ====== Action 参数（可选） ======
        self.action_params = {
            "penalty_scale": LearnableParam(0.05, True, 0.0, 1.0),
            "reward_bias":   LearnableParam(0.00, True, -1.0, 1.0),
        }

    # ====== 网络端口 ======
    def to_tensor(self):
        """
        将所有 learnable 参数转为一个 tensor，用于网络输入或训练。
        """
        vals = []
        for group in [self.sample_params, self.mcts_params, self.action_params]:
            for name, p in group.items():
                if p.learnable:
                    vals.append(p.value)
        return torch.tensor(vals, dtype=torch.float32)

    def update_from_tensor(self, tensor):
        """
        将网络输出的 tensor 写回参数，顺序必须与 to_tensor 对齐。
        """
        idx = 0
        for group in [self.sample_params, self.mcts_params, self.action_params]:
            for name, p in group.items():
                if p.learnable:
                    p.value = float(tensor[idx])
                    p.clamp()
                    idx += 1

    # ====== 系统端口 ======
    def apply_to_system(self, sample_pool, mcts, action_space):
        """
        将当前参数写入到真实系统组件中。
        """
        # 写入 sample pool
        for name, p in self.sample_params.items():
            setattr(sample_pool, name, p.value)

        # 写入 MCTS
        for name, p in self.mcts_params.items():
            setattr(mcts, name, p.value)

        # 写入 Action 类或 Action Manager
        for name, p in self.action_params.items():
            setattr(action_space, name, p.value)

    # ====== 工具函数 ======
    def get_all(self):
        """以 dict 形式打印所有参数，调试/日志用。"""
        out = {}
        for group in [self.sample_params, self.mcts_params, self.action_params]:
            for k, p in group.items():
                out[k] = p.value
        return out