# 特征计算
# 依赖：先执行 research_workflow_lasso_prepare.py

# 过滤Alphalens的warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 加载模块
import polars as pl
from functools import partial

from vnpy.trader.constant import Interval

from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset import (
    AlphaDataset,
    process_drop_na,
    process_cs_norm,
    process_fill_na
)
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158

# 任务参数
name = "300_lasso"
index_symbol: str = "000300.SSE"
start: str = "2008-01-01"
end: str = "2023-12-31"
interval: Interval = Interval.DAILY
extended_days: int = 100

# 创建数据中心
lab: AlphaLab = AlphaLab("./lab/csi300")

# 加载所有成分股代码
component_symbols: list[str] = lab.load_component_symbols(index_symbol, start, end)

# 加载成分股数据
df: pl.DataFrame = lab.load_bar_df(component_symbols, interval, start, end, extended_days)

# 创建数据集对象
dataset: AlphaDataset = Alpha158(
    df,
    train_period=("2008-01-01", "2014-12-31"),
    valid_period=("2015-01-01", "2016-12-31"),
    test_period=("2017-01-01", "2020-8-31"),
)

# 添加数据预处理器
dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))
dataset.add_processor("infer", partial(process_fill_na, fill_value=0))

# 收集指数成分过滤器
filters: dict[str, list[str]] = lab.load_component_filters(index_symbol, start, end)

# 准备特征和标签数据
dataset.prepare_data(filters, max_workers=3)

# 数据预处理
dataset.process_data()

# 特征表现分析
dataset.show_feature_performance("rsv_5")

# 保存到文件缓存（供后续模型训练使用）
lab.save_dataset(name, dataset)
print(f"数据集已保存：{name}")
