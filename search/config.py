class SearchConfig:
    """
    SearchConfig configures the parameters for structured prompt optimization tasks.
    It includes settings for data splitting, search depth, RNN parameters, MCTS control, etc.

    Note: Field names remain unchanged to ensure external compatibility.
    """

    def __init__(self):
        # Dataset split settings
        self.split_ratio: float = 0.7      # (Training + Validation) / Total data
        self.split_ratio_train_val: float = 0.8     # Training / (Training + Validation)
        self.split_ratio_train: float = 0.5  # Train_mcts/ (Train_mcts + Train_rnn)
        self.shuffle_seed: int = 42        # Random seed for data shuffling

        # MCTS search control
        self.exploration_weight: float = 1.5   # Exploration factor in UCT formula
        self.iter_num: int = 50                 # Number of iterations in MCTS main loop
        self.depth_threshold: int = 5          # Maximum search depth
        self.width_threshold: int = 3          # Number of children per expanded node
        self.action_structure_flush_ratio: int = 1

        # Node expansion and rollout config
        self.batch_size: int = 10              # Batch size for training
        self.expand_num: int = 3               # Number of nodes to expand per iteration (for parallelism)
        assert self.expand_num <= self.width_threshold
        self.rollout_length: int = 4           # Rollout path depth (fixed-length simulation)
        self.rollout_width: int = 3
        self.rollout_early_stop_rounds: int = 3
        self.rollout_early_stop_delta: float = 0.01

        # RNN Controller settings
        self.rnn_batch_size: int = 32         # Batch size for RNN training
        self.rnn_hidden_dim: int = 128         # Hidden dimension of RNN
        self.rnn_lr: float = 1e-3              # Learning rate for RNN
        self.rnn_structure_contribution = False

        # Multi-threaded reward evaluation
        self.reward_thread_num: int = 16       # Number of threads for reward evaluation

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