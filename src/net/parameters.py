import torch
import torch.nn as nn
from typing import Dict

class ParamBundle(nn.Module):
    """
    ParamBundle = 被 controller 操作的「环境参数状态」
    controller 永远不直接预测 ParamBundle
    只预测 ΔParamBundle
    """
    """
    可学习参数管理器（只保留学习参数）

    - Informative Score Head: 3 个参数 (diff, gain, var)
    - Sample Pool / Reward 调制: 3 个参数 (easy, informative, hard)
    - MCTS Head: alpha
    """
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        # =====================================
        # Informative Score Head (learnable)
        # =====================================
        self.informative_logits = nn.Parameter(
            torch.zeros(3, dtype=torch.float32, device=device)
        )

        # =====================================
        # Sample Pool / Reward 调制 (learnable)
        # =====================================
        self.pool_logits = nn.Parameter(
            torch.zeros(3, dtype=torch.float32, device=device)
        )

        # =====================================
        # MCTS Head (learnable alpha)
        # =====================================
        self.mcts_alpha = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=device)
        )

    # =====================================================
    # 网络接口
    # =====================================================
    def to_tensor(self) -> torch.Tensor:
        """展平成一个参数向量（供网络预测）"""
        return torch.cat([
            self.informative_logits,
            self.pool_logits,
            self.mcts_alpha.unsqueeze(0)
        ], dim=0)
    
    def apply_delta(self, delta: torch.Tensor, lr: float = 0.1):
        """应用网络预测的参数增量"""
        with torch.no_grad():
            self.informative_logits += lr * delta[0:3]
            self.pool_logits += lr * delta[3:6]
            self.mcts_alpha += lr * delta[6]

    # ===========================
    # getter / 系统接口
    # ===========================
    def get_informative_weights(self):
        """返回 softmax 权重，可直接用于 informative head"""
        return torch.softmax(self.informative_logits, dim=-1)

    def get_pool_weights(self):
        """返回 softmax 权重，可直接用于 pool head"""
        return torch.softmax(self.pool_logits, dim=-1)

    def get_mcts_alpha(self):
        """返回 MCTS alpha 参数 (0~1)"""
        return torch.sigmoid(self.mcts_alpha_logit)

    # =====================================================
    # Debug / Log
    # =====================================================
    def as_dict(self) -> Dict[str, float]:
        names = (
            ["info_diff_logit", "info_gain_logit", "info_var_logit"]
            + ["pool_easy_logit", "pool_informative_logit", "pool_hard_logit"]
            + ["mcts_alpha"]
        )
        return dict(zip(names, self.to_tensor().tolist()))