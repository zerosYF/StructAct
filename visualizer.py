import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better thread safety
import matplotlib.pyplot as plt
import networkx as nx
import hashlib
import threading
import time
from matplotlib.gridspec import GridSpec
import warnings

def hash_str(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:6]

class MCTSVisualizer:
    def __init__(self, mcts=None, root=None, interval=2.0, max_nodes=100):
        self.mcts = mcts
        self.root = root
        self.interval = interval
        self.max_nodes = max_nodes
        self.rewards = []
        self.entropies = []
        self._stop_event = threading.Event()
        self._force_update = False
        self._thread = threading.Thread(target=self._run)

    def start(self, title: str = "MCTS Visualization"):
        self.title = title
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
    
    def set_mcts(self, mcts, root, max_nodes=100):
        self.mcts = mcts
        self.root = root
        self.max_nodes = max_nodes

    def log_train(self, reward: float, entropy:float):
        self.rewards.append(reward)
        self.entropies.append(entropy)
        # Force immediate update for better real-time display
        if hasattr(self, '_force_update'):
            self._force_update = True

    def _run(self):
        # Suppress the GUI thread warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread")
            plt.ion()
            fig = plt.figure(figsize=(12, 8), num=self.title)
        gs = GridSpec(2, 1, height_ratios=[1, 2]) 
        ax_train = fig.add_subplot(gs[0])
        ax_train_ = ax_train.twinx()
        ax_tree = fig.add_subplot(gs[1])

        while not self._stop_event.is_set():
            # Don't clear axes here, let each draw method handle it
            self._draw_train_curve(ax_train, ax_train_)

            if self.mcts and self.root:
                ax_tree.clear()
                self._draw_tree(ax_tree)

            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Use shorter sleep if force update is requested
            if self._force_update:
                time.sleep(0.1)  # Quick update
                self._force_update = False
            else:
                time.sleep(self.interval)

        plt.ioff()
        plt.show()

    def _draw_train_curve(self, ax_l, ax_r):
        # Clear both axes first
        ax_l.clear()
        ax_r.clear()
        
        if not self.rewards:
            ax_l.set_title("Waiting for RNN data...")
            ax_l.set_xlabel("Epoch")
            ax_l.set_ylabel("Reward", color="green")
            return
        
        # Set title and labels
        ax_l.set_title(f"RNN Training Progress (Epoch {len(self.rewards)})")
        ax_l.set_xlabel("Epoch")
        ax_l.set_ylabel("Reward", color="green")
        ax_l.tick_params(axis='y', labelcolor="green")
        
        # Plot data
        x = list(range(1, len(self.rewards) + 1))  # Start from 1 for epoch numbers
        ax_l.plot(x, self.rewards, 'o-', label="Mean Reward", color="green", linewidth=2, markersize=6)
        
        # Configure right y-axis for entropy
        ax_r.set_ylabel("Entropy", color="blue")
        ax_r.tick_params(axis='y', labelcolor="blue")
        ax_r.yaxis.set_label_position("right")
        
        # Plot entropy if available
        if self.entropies and len(self.entropies) == len(self.rewards):
            ax_r.plot(x, self.entropies, 's-', label="Entropy", color="blue", linewidth=2, markersize=6)
        
        # Add grid and legend
        ax_l.grid(True, alpha=0.3)
        lines1, labels1 = ax_l.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax_l.legend(lines1 + lines2, labels1 + labels2, loc="best")
        
        # Add latest values as text
        if self.rewards:
            latest_reward = self.rewards[-1]
            ax_l.text(0.02, 0.98, f"Latest Reward: {latest_reward:.4f}", 
                     transform=ax_l.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if self.entropies:
            latest_entropy = self.entropies[-1]
            ax_r.text(0.98, 0.98, f"Latest Entropy: {latest_entropy:.4f}", 
                     transform=ax_r.transAxes, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    def _draw_tree(self, ax):
        G = nx.DiGraph()
        labels = {}
        queue = [self.root]
        visited = set()

        while queue and len(G.nodes) < self.max_nodes:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            G.add_node(node)

            q = self.mcts.Q.get(node, 0)
            n = self.mcts.N.get(node, 0)
            qn_ratio = f"{(q / n):.2f}" if n else "?"
            action_name = node.action_seq[-1].name if node.action_seq else "Root"
            labels[node] = f"{action_name}\nQ={q:.4f}, N={n}, Q/N={qn_ratio}, reward={node.reward_value:.2f}"

            children = self.mcts.children.get(node, [])
            for child in children:
                G.add_edge(node, child)
                queue.append(child)

        if len(G.nodes) == 0:
            return

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')  
        except Exception:
            pos = nx.spring_layout(G, seed=42)  

        nx.draw(
            G, pos, with_labels=False, node_color='lightblue',
            node_size=1400, edge_color='gray', arrows=True, ax=ax
        )
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        ax.set_title("MCTS")
        ax.axis('off')

Visualizer = MCTSVisualizer()