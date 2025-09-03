class SearchConfig:
    """
    SearchConfig configures the parameters for structured prompt optimization tasks.
    It includes settings for data splitting, search depth, RNN parameters, MCTS control, etc.

    Note: Field names remain unchanged to ensure external compatibility.
    """

    def __init__(self):
        self.shuffle_seed: int = 42        # Random seed for data shuffling
        # MCTS search control
        self.exploration_weight: float = 1.5   # Exploration factor in UCT formula
        self.mcts_iter_num_max: int = 10          # Number of iterations in MCTS main loop
        self.mcts_iter_num_min: int = 0           # Minimum MCTS iterations
        self.depth_threshold: int = 5          # Maximum search depth
        self.width_threshold: int = 3          # Number of children per expanded node
        # Node expansion and rollout config
        self.batch_size: int = 10              # Batch size for training
        self.expand_num_min: int = 0               # Number of nodes to expand per iteration (for parallelism)
        self.expand_num_max: int = 3               # Maximum number of nodes to expand
        assert self.expand_num_max <= self.width_threshold
        self.rollout_length_min: int = 1          # Minimum rollout path depth
        self.rollout_length_max: int = 4           # Rollout path depth (fixed-length simulation)
        self.rollout_width: int = 1
        self.rollout_early_stop_rounds: int = 3
        self.rollout_early_stop_delta: float = 0.01

        # RNN Controller settings
        self.rnn_iter_num: int = 400        # Number of training iterations for RNN
        self.rnn_hidden_dim: int = 128         # Hidden dimension of RNN
        self.rnn_lr: float = 3e-4              # Learning rate for RNN (increased from 1e-4)
        self.rnn_rl_reward_scale = 10          # Reward scale (reduced from 100 to avoid large value loss)

        # Multi-threaded reward evaluation
        self.reward_thread_num: int = 8       # Number of threads for reward evaluation

        # Debugging indicators (for tracing internal states)
        self.rollout_idx: int = 1              # Current rollout path index
        self.choose_idx: int = 2               # Current chosen action index
        self.uct_idx: int = 0                  # Current UCT path index
        self.model_idx: int = 0                # Current API model index

        # Model configuration
        self.model_name: str = "qwen3-30b-a3b-instruct-2507" # zhiyan3
        self.api_key: str =  "62507c5f747745ada8bbd35e570788d2" # "12a83368c59545e6af1c0a17810cd675" # "zhiyan123"
        self.base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1" # "https://api.deepseek.com/v1" # "http://192.168.200.222:12025/v1"

        # Ollama model configuration
        self.ollama_model_name: str = "llama3.1:8b"          # Name of the Ollama model
        self.ollama_base_url: str = "http://localhost:11434"  # Base URL for local Ollama server

        self.eval_model_name: str = "gpt-3.5-turbo"
        self.eval_api_key: str = ""

        self.optim_model_name:str = "gpt-4"
        self.optim_api_key: str = ""