class SearchConfig:
    """
    SearchConfig configures the parameters for structured prompt optimization tasks.
    It includes settings for data splitting, search depth, RNN parameters, MCTS control, etc.

    Note: Field names remain unchanged to ensure external compatibility.
    """

    def __init__(self):
        self.shuffle_seed: int = 42        # Random seed for data shuffling
        # MCTS search control
        self.exploration_weight: float = 2.5   # Exploration factor in UCT formula
        self.mcts_iter_num_max: int = 10         # Number of iterations in MCTS main loop
        self.mcts_iter_num_min: int = 0           # Minimum MCTS iterations
        self.max_depth_threshold: int = 5          # Maximum search depth
        self.min_depth_threshold: int = 3

        self.width_threshold: int = 3          # Number of children per expanded node
        self.rollout_threshold:int = 3
        # Node expansion and rollout config
        self.batch_size: int = 5              # Batch size for training

        # RNN Controller settings
        self.rnn_iter_num: int = 400        # Number of training iterations for RNN
        self.rnn_hidden_dim: int = 128         # Hidden dimension of RNN
        self.rnn_lr: float = 3e-4              # Learning rate for RNN (increased from 1e-4)
        self.rnn_rl_reward_scale = 10          # Reward scale (reduced from 100 to avoid large value loss)

        # Multi-threaded reward evaluation
        self.reward_thread_num: int = 3       # Number of threads for reward evaluation

        # Debugging indicators (for tracing internal states)
        self.rollout_idx: int = 1              # Current rollout path index
        self.choose_idx: int = 2               # Current chosen action index

        self.use_pool: bool = False            # Whether to use a sample pool for rollouts

        # Model configuration
        # self.model_name: str = "zhiyan3"
        # self.api_key: str =  "zhiyan123" 
        # self.base_url: str =  "http://192.168.200.222:12025/v1"

        self.eval_model_name: str = "qwen-flash"
        self.eval_api_key: str = "62507c5f747745ada8bbd35e570788d2"
        self.eval_model_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.eval_model_temperature: float = 0.0

        self.optim_model_name:str = "deepseek-chat"
        self.optim_api_key: str = "12a83368c59545e6af1c0a17810cd675"
        self.optim_model_url:str = "https://api.deepseek.com/v1"
        self.optim_model_temperature: float = 1.0