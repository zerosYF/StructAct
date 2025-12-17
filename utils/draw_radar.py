import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# 整合所有任务 & 数据
# ------------------------
tasks = [
    # group1
    "Causal Judgement(BBH)", "Epistemic(BBH)", "Geometric Shapes(BBH)", "Object Counting(BBH)", "Penguins Table(BBH)",
    # group2
    "Pro Engineering", "Pro Business", "Multi label Ethos", "MedQA", "CaseHold",
    # group3
    "Casual Judge(BBEH)", "Temporal Sequences(BBEH)", "Object Counting(BBEH)", "Boolean Expressions(BBEH)"
]

methods = {
    "HumanZS": [0.53,0.65,0.42,0.596,0.556, 0.485,0.47,0.16,0.61,0.64, 0.425,0.0125,0,0.3375],
    "HumanFS": [0.57,0.74,0.45,0.654,0.683, 0.51,0.47,0.225,0.685,0.67, 0.4625,0.0125,0,0.2375],
    "HumanFS-Analytic": [0.49,0.78,0.45,0.472,0.708, 0.46,0.71,0.275,0.755,0.655, 0.5875,0.0375,0.0375,0.175],
    "CoTZS": [0.57,0.836,0.895,0.916,0.924, 0.72,0.795,0.185,0.765,0.67, 0.525,0.2625,0.175,0.475],
    "CoTFS": [0.65,0.818,0.865,0.904,0.924, 0.725,0.86,0.295,0.795,0.71, 0.575,0.2375,0.2125,0.375],
    "CoTFS-Analytic": [0.62,0.858,0.845,0.866,0.848, 0.745,0.855,0.395,0.815,0.71, 0.55,0.1625,0.175,0.3625],
    "PromptAgent": [0.67,0.896,0.86,0.912,0.974, 0.755,0.815,0.32,0.795,0.72, 0.5375,0.2,0.1875,0.3875],
    "Our": [0.68,0.954,0.91,0.986,0.974, 0.75,0.875,0.435,0.82,0.745, 0.4625,0.225,0.3,0.525],
}

# ------------------------
# 画雷达图
# ------------------------
N = len(tasks)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for method, values in methods.items():
    vals = values + values[:1]
    if method == "Our":
        ax.plot(angles, vals, label=method, linewidth=3.5, color="red")  # 仅突出我们的方法
    else:
        ax.plot(angles, vals, label=method, linewidth=1)

# 设置角度标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(tasks, fontsize=11, rotation=20)

ax.set_yticklabels([])
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig("main.png", dpi=300, bbox_inches='tight')  # 保存为 PNG，300dpi
plt.show()