import json
from collections import defaultdict
import os

# 输入文件路径（大json文件，每条是一条dict）
input_file = "dataset/bbeh/bbeh_data.json"
output_dir = "dataset/bbeh/bbeh_by_task"

os.makedirs(output_dir, exist_ok=True)

# 读取整个列表
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)  # 直接 load 列表

# 按 task 分组
task_dict = defaultdict(list)
for item in data:
    task = item.get("task", "unknown")
    task_dict[task].append(item)

# 写出到多个文件（每个文件还是 list 格式）
for task, items in task_dict.items():
    safe_task = task.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(output_dir, f"{safe_task}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(items)} samples to {out_path}")