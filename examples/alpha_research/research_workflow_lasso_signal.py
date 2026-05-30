# 预测信号
# 依赖：先执行 research_workflow_lasso_feature.py 和 research_workflow_lasso_train.py

# 加载模块
import numpy as np
import polars as pl

from vnpy.alpha import Segment, AlphaDataset, AlphaModel, AlphaLab

# 任务参数
name = "300_lasso"

# 创建数据中心
lab: AlphaLab = AlphaLab("./lab/csi300")

# 从文件缓存加载数据集
dataset: AlphaDataset = lab.load_dataset(name)

# 加载模型
model: AlphaModel = lab.load_model(name)

# 用模型在测试集上预测
pre: np.ndarray = model.predict(dataset, Segment.TEST)

# 加载测试集数据
df_t: pl.DataFrame = dataset.fetch_infer(Segment.TEST)

# 合并预测信号列
df_t = df_t.with_columns(pl.Series(pre).alias("signal"))

# 提取信号数据
signal: pl.DataFrame = df_t["datetime", "vt_symbol", "signal"]

# 检查信号绩效
dataset.show_signal_performance(signal)

# 保存信号数据（供后续策略回测使用）
lab.save_signal(name, signal)
print(f"信号已保存：{name}")
