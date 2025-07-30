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
        self.rnn_iter_num: int = 500        # Number of training iterations for RNN
        self.rnn_hidden_dim: int = 128         # Hidden dimension of RNN
        self.rnn_lr: float = 1e-4              # Learning rate for RNN
        self.rnn_rl_reward_scale = 100
        self.rnn_rl_baseline:float = 0.91
        self.rnn_rl_baseline_alpha = 0.5
        self.rnn_rl_min_entropy_weight = 0.001
        self.rnn_rl_max_entropy_weight = 0.5
        self.rnn_rl_entropy_decay_rate = 0.95

        self.rnn_structure_contribution = False
        self.rnn_attribution_interval = 10
        self.rnn_aux_loss_coef=1.5

        # Multi-threaded reward evaluation
        self.reward_thread_num: int = 40       # Number of threads for reward evaluation

        # Debugging indicators (for tracing internal states)
        self.rollout_idx: int = 1              # Current rollout path index
        self.choose_idx: int = 2               # Current chosen action index
        self.uct_idx: int = 0                  # Current UCT path index
        self.model_idx: int = 0                # Current API model index

        # Model configuration
        self.model_name: str = "zhiyan3"
        self.api_key: str = "zhiyan123"
        self.base_url: str = "http://192.168.200.222:12025/v1"

        # Ollama model configuration
        self.ollama_model_name: str = "llama3.1:8b"          # Name of the Ollama model
        self.ollama_base_url: str = "http://localhost:11434"  # Base URL for local Ollama server

        self.eval_model_name: str = "gpt-3.5-turbo"
        self.eval_api_key: str = ""

        self.optim_model_name:str = "gpt-4"
        self.optim_api_key: str = ""