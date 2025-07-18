class SearchConfig:
    """
    SearchConfig 用于配置结构化提示词优化任务中的搜索参数。
    包含数据划分比例、搜索深度、RNN 超参数、MCTS 相关参数等。

    注意：字段名保持不变，以保证外部兼容。
    """

    def __init__(self):
        # 数据集划分设置
        self.split_ratio: float = 0.7      # (训练集 + 验证集) / 总数据
        self.split_ratio_: float = 0.8     # 训练集 / (训练集 + 验证集)
        self.shuffle_seed: int = 42        # 数据打乱的随机种子

        # MCTS 搜索控制
        self.exploration_weight: float = 1.5   # UCT 中的探索因子
        self.iter_num: int = 5                # MCTS 主循环迭代次数
        self.depth_threshold: int = 4          # 搜索最大深度

        # 节点拓展与 rollout 配置
        self.batch_size: int = 10              # 每轮可扩展节点数量（用于并行）
        self.expand_num: int = 3               # 每个节点拓展出多少子节点
        self.rollout_parallel: bool = True     # 是否并行执行 rollout
        self.rollout_length: int = 3           # rollout 路径深度（定长模拟）
        self.rollout_path_num: int = 3         # 每次 rollout 的路径数量

        # RNN Controller 设置
        self.rnn_hidden_dim: int = 128         # RNN 隐藏层维度
        self.rnn_lr: float = 1e-3              # RNN 学习率

        # 多线程评估配置
        self.reward_thread_num: int = 16       # reward 评估并行线程数

        # 调试标记（可用于分析）
        self.rollout_idx: int = 0              # 当前 rollout 路径编号
        self.choose_idx: int = 0               # 当前选择动作编号
        self.uct_idx: int = 0                  # 当前 UCT 路径编号

         # 模型配置
        self.model_name: str = "zhiyan3"
        self.api_key: str = "zhiyan123"
        self.base_url: str = "http://192.168.200.222:12025/v1"

        # Ollama 模型配置
        self.ollama_model_name: str = "llama3.1:8b"     # Ollama 模型名称
        self.ollama_base_url: str = "http://localhost:11434"  # Ollama 本地服务地址