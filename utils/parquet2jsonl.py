import pandas as pd

# 输入 Parquet 文件路径
parquet_file = "combi1.parquet"  # 替换为你的文件路径
# 输出 JSONL 文件路径
jsonl_file = "combi1.jsonl"

# 读取 Parquet 文件
df = pd.read_parquet(parquet_file)

# 导出为 JSONL，每行一个 JSON 对象
df.to_json(jsonl_file, orient='records', lines=True, force_ascii=False)

print(f"已成功将 {parquet_file} 转换为 {jsonl_file}")