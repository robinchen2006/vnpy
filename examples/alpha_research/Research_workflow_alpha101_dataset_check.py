"""
模型训练前诊断脚本
检查三个核心疑点：
  1. train/valid 是否按时间顺序切分（有无交叉）
  2. label 的时间偏移是否正确（有无未来数据泄露）
  3. train/valid 市场环境差异是否过大
"""

import os
import random

import numpy as np
import polars as pl

from vnpy.alpha import AlphaDataset, AlphaLab, Segment

# ── 任务参数 ──────────────────────────────────────────────
name = "300_lgb"

current_folder = os.path.dirname(os.path.abspath(__file__))
csi300_root = os.path.join(current_folder, "./lab/csi300")

lab: AlphaLab = AlphaLab(csi300_root)
dataset: AlphaDataset = lab.load_dataset(name)


# ════════════════════════════════════════════════════════════
# 诊断 1：train / valid 时间范围是否严格不重叠
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【诊断 1】train / valid 时间切分检查")
print("=" * 60)

train_df: pl.DataFrame = dataset.fetch_learn(Segment.TRAIN)
valid_df: pl.DataFrame = dataset.fetch_learn(Segment.VALID)

train_dates = train_df["datetime"].unique().sort()
valid_dates = valid_df["datetime"].unique().sort()

train_start, train_end = train_dates[0], train_dates[-1]
valid_start, valid_end = valid_dates[0], valid_dates[-1]

print(f"训练集：{train_start} → {train_end}，共 {train_dates.len()} 个交易日")
print(f"验证集：{valid_start} → {valid_end}，共 {valid_dates.len()} 个交易日")

overlap = set(train_dates.to_list()) & set(valid_dates.to_list())
if overlap:
    print(f"\n⚠️  [严重] 发现 {len(overlap)} 个重叠日期！存在数据泄露风险。")
    print(f"   重叠日期示例（前5个）: {sorted(overlap)[:5]}")
else:
    print("\n✅  日期无重叠")

if train_end >= valid_start:
    print(
        f"⚠️  [严重] 训练集结束日 {train_end} >= 验证集开始日 {valid_start}，时间顺序错误！"
    )
else:
    gap_days = (valid_start - train_end).days
    print(f"✅  时间顺序正确，间隔 {gap_days} 天")


# ════════════════════════════════════════════════════════════
# 诊断 2：label 的时间偏移是否正确
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【诊断 2】label 时间偏移检查（检测未来数据泄露）")
print("=" * 60)

# fetch_learn 只包含 alpha 因子列和 label，没有原始价格列
# 打印单只股票的连续 label 序列，确认数值范围符合"未来收益"的预期
sample_symbol = train_df["vt_symbol"][0]
sample = (
    train_df.filter(pl.col("vt_symbol") == sample_symbol)
    .sort("datetime")
    .select(["datetime", "vt_symbol", "label"])
    .head(10)
)
print(f"\n{sample_symbol} 前10行 label（正常日收益应在 -0.1 ~ 0.1 之间）：")
print(sample)

# ── 检验 2-A：label 自相关（lag-1）────────────────────────
# 正确的 forward return：label[t] = close[t+N]/close[t] - 1
# 若 label 用了当日收益，相邻两天同一股票的 label 会出现"负自相关"
# 若 label 是未来某天的收益，与当日 label 的相关性应接近 0
lag_df = (
    train_df.sort(["vt_symbol", "datetime"])
    .with_columns(pl.col("label").shift(1).over("vt_symbol").alias("label_lag1"))
    .drop_nulls(["label", "label_lag1"])
)

autocorr = lag_df.select(
    pl.corr("label", "label_lag1", method="spearman").alias("ac")
).item()

print(f"\n【2-A】label 一阶自相关（lag-1 Spearman）：{autocorr:.4f}")
if autocorr > 0.8:
    print(
        "⚠️  [严重] 自相关极高（>0.8），label 可能几乎没有偏移，或使用了累积收益而非单期收益。"
    )
elif autocorr > 0.3:
    print(
        "⚠️  [注意] 自相关偏高，label 可能是多日累积收益（如 5日/20日），请确认是否符合预期。"
    )
elif autocorr < -0.3:
    print("⚠️  [注意] 自相关为负，label 可能是当日收益（均值回归），存在泄露风险。")
else:
    print("✅  自相关接近 0，符合单期 forward return 的预期特征。")

# ── 检验 2-B：label 与各 alpha 因子的截面 IC ──────────────
# 正常情况下单个 alpha 因子与 label 的 IC 应在 ±0.10 以内
# 若某因子 IC 异常高（>0.3），可能存在因子与 label 同期计算的泄露
factor_cols = [c for c in train_df.columns if c.startswith("alpha")]

ic_list = []
sample_dates = train_df["datetime"].unique().sort().to_list()
# 随机抽取最多 200 个交易日避免计算过慢
random.seed(42)
sample_dates_subset = random.sample(sample_dates, min(200, len(sample_dates)))

for col in factor_cols:
    daily_ics = []
    for dt in sample_dates_subset:
        day_df = (
            train_df.filter(pl.col("datetime") == dt)
            .select([col, "label"])
            .drop_nulls()
        )
        if len(day_df) < 10:
            continue
        ic_val = day_df.select(pl.corr(col, "label", method="spearman")).item()
        if ic_val is not None and not np.isnan(ic_val):
            daily_ics.append(ic_val)
    if daily_ics:
        ic_list.append(
            {
                "factor": col,
                "mean_ic": float(np.mean(daily_ics)),
                "ic_std": float(np.std(daily_ics)),
            }
        )

ic_df = pl.DataFrame(ic_list).sort("mean_ic", descending=True)

print(
    f"\n【2-B】各 alpha 因子与 label 的 Mean Rank IC（基于 {len(sample_dates_subset)} 个交易日抽样）："
)
print(f"  IC > 0 的因子数：{(ic_df['mean_ic'] > 0).sum()} / {len(ic_df)}")
print("  IC 绝对值最大的前5个因子：")
top5 = (
    ic_df.with_columns(pl.col("mean_ic").abs().alias("abs_ic"))
    .sort("abs_ic", descending=True)
    .head(5)
)
for row in top5.iter_rows(named=True):
    flag = "⚠️  异常高" if abs(row["mean_ic"]) > 0.15 else "      "
    print(
        f"  {flag}  {row['factor']:10s}  mean_ic={row['mean_ic']:+.4f}  ic_std={row['ic_std']:.4f}"
    )

max_abs_ic = top5["abs_ic"][0]
if max_abs_ic > 0.3:
    print(
        "\n⚠️  [严重] 存在因子 IC > 0.3，极不正常，强烈怀疑因子与 label 存在同期计算泄露。"
    )
elif max_abs_ic > 0.15:
    print(
        "\n⚠️  [注意] 存在因子 IC 偏高（>0.15），请核查该因子的计算是否引用了未来数据。"
    )
else:
    print("\n✅  所有因子 IC 在合理范围内（< 0.15）。")

# ── label 分布概览 ────────────────────────────────────────
label_stats = train_df["label"].describe()
print("\n【2-C】label 统计（训练集）：")
print(label_stats)

label_std = train_df["label"].std()
print(f"标准差：{label_std:.6f}")

if label_std < 1e-4:
    print("⚠️  [严重] label 标准差极小，数据几乎无变化，请检查 label 构造逻辑。")
elif label_std > 0.5:
    print(
        "⚠️  [注意] label 标准差较大（>0.5），建议对 label 做截面 rank 归一化后再训练。"
    )
else:
    print("✅  label 分布基本正常")


# ════════════════════════════════════════════════════════════
# 诊断 3：train / valid 市场环境差异
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【诊断 3】train / valid 市场环境差异检查")
print("=" * 60)

for seg_name, df in [("train", train_df), ("valid", valid_df)]:
    label_mean = df["label"].mean()
    label_std = df["label"].std()
    label_min = df["label"].min()
    label_max = df["label"].max()
    n_symbols = df["vt_symbol"].n_unique()
    n_dates = df["datetime"].n_unique()
    print(
        f"\n{seg_name}：股票数={n_symbols}，交易日={n_dates}，"
        f"label 均值={label_mean:.5f}，std={label_std:.5f}，"
        f"min={label_min:.5f}，max={label_max:.5f}"
    )

daily_label = (
    pl.concat(
        [
            train_df.with_columns(pl.lit("train").alias("segment")),
            valid_df.with_columns(pl.lit("valid").alias("segment")),
        ]
    )
    .group_by(["datetime", "segment"])
    .agg(pl.col("label").mean().alias("cross_mean"))
    .sort("datetime")
)

train_cross_mean = daily_label.filter(pl.col("segment") == "train")["cross_mean"].mean()
valid_cross_mean = daily_label.filter(pl.col("segment") == "valid")["cross_mean"].mean()

print(f"\n训练期日均截面 label 均值：{train_cross_mean:.5f}")
print(f"验证期日均截面 label 均值：{valid_cross_mean:.5f}")

diff = abs(train_cross_mean - valid_cross_mean)
if diff > 0.005:
    print(
        f"⚠️  [注意] 两段市场均值差异 {diff:.5f} 较大，验证期市场环境与训练期差异显著（可能是牛熊转换）。"
    )
else:
    print(f"✅  两段市场环境差异可接受（均值差 {diff:.5f}）")

# 打印 valid 各年度的平均 label，便于识别哪一年是问题所在
print("\n验证期各年度 label 均值（识别问题年份）：")
valid_yearly = (
    valid_df.with_columns(pl.col("datetime").dt.year().alias("year"))
    .group_by("year")
    .agg(
        pl.col("label").mean().alias("mean_label"),
        pl.col("label").std().alias("std_label"),
    )
    .sort("year")
)
print(valid_yearly)
