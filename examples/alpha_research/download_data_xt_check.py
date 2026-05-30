# 列出 daily 目录下的 parquet 文件信息
import os
from pathlib import Path

import polars as pl

from vnpy.alpha import AlphaLab

# 与 download_data_xt.py 保持一致
task_name = "csi300"
current_folder = os.path.dirname(os.path.abspath(__file__))
task_path = os.path.join(current_folder, f"./lab/{task_name}")

lab = AlphaLab(task_path)    # 指定数据文件夹")

daily_path: Path = lab.daily_path
files = sorted(daily_path.glob("*.parquet"))

print(f"{'文件名':<30} {'行数':>8} {'开始日期':>12} {'结束日期':>12} {'文件大小(KB)':>14}")
print("-" * 82)

for file in files:
    df = pl.read_parquet(file)
    rows = len(df)
    size_kb = file.stat().st_size / 1024
    if rows > 0:
        start_dt = df["datetime"].min()
        end_dt = df["datetime"].max()
        start_str = str(start_dt)[:10]
        end_str = str(end_dt)[:10]
    else:
        start_str = end_str = "N/A"
    print(f"{file.name:<30} {rows:>8} {start_str:>12} {end_str:>12} {size_kb:>14.1f}")

print("-" * 82)
print(f"共 {len(files)} 个文件")
