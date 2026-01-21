import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better thread safety
import matplotlib.pyplot as plt
import networkx as nx
import hashlib
import threading
import time
from matplotlib.gridspec import GridSpec
import numpy as np
from src.net.parameters import ParamBundle

def hash_str(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:6]

class UnifiedVisualizer:
    def __init__(self, root=None, interval=2.0, max_nodes=100, max_children=5):
        self.root = root
        self.interval = interval
        self.max_nodes = max_nodes
        self.max_children = max_children  # 每层最多展开的子节点
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run)

        self.info_weights = []
        self.pool_weights = []
        self.alphas = []

        self.info_entropy = []
        self.pool_entropy = []
        self.delta_norms = []

    def set_root(self, root, max_nodes=100):
        self.root = root
        self.max_nodes = max_nodes
    
    def log(self, bundle: ParamBundle):
        info_w = bundle.get_informative_weights().detach().cpu().numpy()
        pool_w = bundle.get_pool_weights().detach().cpu().numpy()
        alpha = bundle.mcts_alpha.detach().cpu().item()

        self.info_weights.append(info_w)
        self.pool_weights.append(pool_w)
        self.alphas.append(alpha)

        # ---------- 熵 ----------
        def entropy(p):
            return -np.sum(p * np.log(p + 1e-8))

        self.info_entropy.append(entropy(info_w))
        self.pool_entropy.append(entropy(pool_w))

        # ---------- 参数变化幅度 ----------
        if len(self.info_weights) > 1:
            prev = np.concatenate([
                self.info_weights[-2],
                self.pool_weights[-2],
                [self.alphas[-2]]
            ])
            curr = np.concatenate([
                info_w,
                pool_w,
                [alpha]
            ])
            self.delta_norms.append(np.linalg.norm(curr - prev))
        else:
            self.delta_norms.append(0.0)


    def start(self, title="MCTS Visualization"):
        self.title = title
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _run(self):
        plt.ion()
        fig = plt.figure(figsize=(12, 8), num=self.title)
        gs = GridSpec(2, 1, height_ratios=[1, 2]) 
        ax_train = fig.add_subplot(gs[0])
        ax_train_ = ax_train.twinx()
        ax_tree = fig.add_subplot(gs[1])

        while not self._stop_event.is_set():
            ax_train.clear()
            self._draw_train_curve(ax_train, ax_train_)

            if self.root:
                self._draw_tree(ax_tree, self.root)

            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(self.interval)

        plt.ioff()
        plt.show()
    
    def _draw_train_curve(self, ax_l, ax_r):
        if not self.info_weights:
            ax_l.set_title("waiting for controller data...")
            return

        x = np.arange(len(self.info_weights))

        ax_l.clear()
        ax_l.set_title("Controller Convergence Monitor")
        ax_l.set_xlabel("Step")

        # ===== 左轴：参数 =====
        info = np.array(self.info_weights)
        pool = np.array(self.pool_weights)

        ax_l.plot(x, info[:, 0], label="info_diff")
        ax_l.plot(x, info[:, 1], label="info_gain")
        ax_l.plot(x, info[:, 2], label="info_var")

        ax_l.plot(x, pool[:, 0], "--", label="pool_easy")
        ax_l.plot(x, pool[:, 1], "--", label="pool_info")
        ax_l.plot(x, pool[:, 2], "--", label="pool_hard")

        ax_l.plot(x, self.alphas, "-.", label="mcts_alpha", linewidth=2)

        ax_l.set_ylabel("Parameter Value")
        ax_l.legend(loc="upper left", fontsize=8)

        # ===== 右轴：熵 & Δθ =====
        ax_r.cla()
        ax_r.set_ylabel("Entropy / Δθ")

        ax_r.plot(x, self.info_entropy, color="purple", label="info_entropy")
        ax_r.plot(x, self.pool_entropy, color="brown", label="pool_entropy")
        ax_r.plot(x, self.delta_norms, color="black", label="Δθ_norm")

        ax_r.legend(loc="upper right", fontsize=8)

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
