import matplotlib.pyplot as plt
import networkx as nx
import hashlib
import threading
import time
from matplotlib.gridspec import GridSpec

def hash_str(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:6]

class MCTSVisualizer:
    def __init__(self, mcts=None, root=None, interval=2.0, max_nodes=100):
        self.mcts = mcts
        self.root = root
        self.interval = interval
        self.max_nodes = max_nodes
        self.losses = []
        self.rewards = []
        self.entropies = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
    
    def set_mcts(self, mcts, root, max_nodes=100):
        self.mcts = mcts
        self.root = root
        self.max_nodes = max_nodes

    def log_train(self, loss: float, reward: float, entropy:float):
        """供外部调用，更新RNN训练曲线"""
        self.losses.append(loss)
        self.rewards.append(reward)
        self.entropies.append(entropy)

    def _run(self):
        plt.ion()
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 1, height_ratios=[1, 2])  # 上1行：训练图，下2行：树结构
        ax_train = fig.add_subplot(gs[0])
        ax_train_ = ax_train.twinx()
        ax_tree = fig.add_subplot(gs[1])

        while not self._stop_event.is_set():
            ax_train.clear()
            self._draw_train_curve(ax_train, ax_train_)

            ax_tree.clear()
            self._draw_tree(ax_tree)

            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(self.interval)

        plt.ioff()
        plt.show()

    def _draw_train_curve(self, ax_l, ax_r):
        if not self.losses:
            ax_l.set_title("waiting for rnn data...")
            return
        else:
            ax_l.set_title("RNN")

        x = list(range(len(self.losses)))
        ax_l.set_xlabel("Step")
        # 主轴清除
        ax_l.clear()
        ax_l.set_ylabel("Loss", color="red")
        ax_l.tick_params(axis='y', labelcolor='red')
        ax_l.plot(x, self.losses, label="Loss", color="red")

        # 副轴：清空+绘制 reward
        ax_r.cla()
        ax_r.yaxis.set_label_position("right")
        ax_r.set_ylabel("Reward / Entropy", color="green")
        ax_r.tick_params(axis='y', labelcolor='green')
        ax_r.plot(x, self.rewards, label="Reward", color="green")
        ax_r.plot(x, self.entropies, label="Entropy", color="blue")

        # 合并图例到右上角，挂在右轴上
        lines, labels = ax_l.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax_r.legend(lines + lines2, labels + labels2, loc="upper right")
        ax_l.grid(True)

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
            labels[node] = f"{action_name}\nQ={q:.2f}, N={n}, Q/N={qn_ratio}"

            children = self.mcts.children.get(node, [])
            for child in children:
                G.add_edge(node, child)
                queue.append(child)

        if len(G.nodes) == 0:
            ax.set_title("无可视化节点")
            return

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')  # 层次布局
        except Exception:
            pos = nx.spring_layout(G, seed=42)  # 回退方案

        nx.draw(
            G, pos, with_labels=False, node_color='lightblue',
            node_size=1400, edge_color='gray', arrows=True, ax=ax
        )
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        ax.set_title("MCTS")
        ax.axis('off')

Visualizer = MCTSVisualizer()