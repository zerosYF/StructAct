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

class RNNVisualizer:
    def __init__(self, interval=2.0):
        self.interval = interval
        self.rewards = []
        self.entropies = []
        self._stop_event = threading.Event()
        self._force_update = False
        self._thread = threading.Thread(target=self._run)

    def start(self, title="RNN Training Progress"):
        self.title = title
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def log_train(self, reward: float, entropy: float):
        self.rewards.append(reward)
        self.entropies.append(entropy)
        self._force_update = True

    def _run(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread")
            plt.ion()
            fig = plt.figure(figsize=(8, 5), num=self.title)

        ax = fig.add_subplot(111)
        ax2 = ax.twinx()

        while not self._stop_event.is_set():
            self._draw(ax, ax2)
            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()

            if self._force_update:
                time.sleep(0.1)
                self._force_update = False
            else:
                time.sleep(self.interval)

        plt.ioff()
        plt.show()

    def _draw(self, ax, ax2):
        ax.clear()
        ax2.clear()

        if not self.rewards:
            ax.set_title("Waiting for RNN data...")
            return

        x = list(range(1, len(self.rewards) + 1))
        ax.set_title(f"RNN Training (Epoch {len(self.rewards)})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Reward", color="green")
        ax.plot(x, self.rewards, "o-", color="green", label="Reward")
        ax.tick_params(axis="y", labelcolor="green")

        ax2.set_ylabel("Entropy", color="blue")
        ax2.plot(x, self.entropies, "s-", color="blue", label="Entropy")
        ax2.tick_params(axis="y", labelcolor="blue")

        ax.grid(True, alpha=0.3)

class MCTSVisualizer:
    def __init__(self, root=None, interval=2.0, max_nodes=100, max_children=5):
        self.root = root
        self.interval = interval
        self.max_nodes = max_nodes
        self.max_children = max_children  # 每层最多展开的子节点
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run)

    def set_root(self, root, max_nodes=100):
        self.root = root
        self.max_nodes = max_nodes

    def start(self, title="MCTS Visualization"):
        self.title = title
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _run(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread")
            plt.ion()
            fig = plt.figure(figsize=(14, 9), num=self.title)
        ax = fig.add_subplot(111)

        while not self._stop_event.is_set():
            ax.clear()
            if self.root:
                self._draw_tree(ax, self.root)

            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(self.interval)

        plt.ioff()
        plt.show()

    def _draw_tree(self, ax, root):
        G = nx.DiGraph()
        labels = {}
        queue = [root]
        visited = set()

        while queue and len(G.nodes) < self.max_nodes:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            G.add_node(node)
            q, n, uct = node.Q, node.N, node.uct_value
            qn_ratio = f"{(q / n):.2f}" if n else "?"
            action_name = node.action_seq[-1].name if getattr(node, "action_seq", None) else "Root"
            labels[node] = f"{action_name}\nQ={q:.2f}, N={n} \n UCT={uct:.2f} \n R={node.reward_value:.2f}"

            # 限制展开的子节点数
            children_sorted = sorted(node.children, key=lambda c: c.Q / (c.N + 1e-6), reverse=True)
            for child in children_sorted[:self.max_children]:
                G.add_edge(node, child)
                queue.append(child)

        if not G.nodes:
            return

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  
        except Exception:
            pos = self._hierarchy_pos(G, root)  

        nx.draw(
            G, pos, with_labels=False, node_color="lightblue",
            node_size=1400, edge_color="gray", arrows=True, ax=ax
        )
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        ax.set_title("MCTS Search Tree")
        ax.axis("off")

    def _hierarchy_pos(self, G, root, width=1., vert_gap=0.3, vert_loc=0, xcenter=0.5):
        """
        如果没安装 graphviz，则使用递归分层布局
        """
        def _hierarchy_pos(G, root, left, right, vert_loc, pos):
            pos[root] = ((left + right) / 2, vert_loc)
            children = list(G.successors(root))
            if len(children) != 0:
                dx = (right - left) / len(children)
                nextx = left
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G, child, nextx - dx, nextx, vert_loc - vert_gap, pos)
            return pos

        return _hierarchy_pos(G, root, 0, width, vert_loc, {})
