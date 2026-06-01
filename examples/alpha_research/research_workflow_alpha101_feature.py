# 特征计算
# 依赖：先执行 research_workflow_alpha101_prepare.py

# 过滤Alphalens的warning


import os
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
    process_cs_norm
)
from vnpy.alpha.dataset.datasets.alpha_101 import Alpha101

def main():
    # 任务参数
    name = "300_lgb"
    index_symbol: str = "000300.SSE"
    start: str = "2008-01-01"
    end: str = "2025-12-31"
    interval: Interval = Interval.DAILY
    extended_days: int = 100

    # 创建数据中心
    current_folder = os.path.dirname(os.path.abspath(__file__))
    csi300_root = os.path.join(current_folder, f"./lab/csi300")

    lab: AlphaLab = AlphaLab(csi300_root)

    # 加载所有成分股代码
    component_symbols: list[str] = lab.load_component_symbols(index_symbol, start, end)

    # 加载成分股数据
    df: pl.DataFrame = lab.load_bar_df(component_symbols, interval, start, end, extended_days)

    # 创建数据集对象
    dataset: AlphaDataset = Alpha101(
        df,
        train_period=("2008-01-01", "2020-12-31"),
        valid_period=("2021-01-01", "2023-12-31"),
        test_period=("2024-01-01", "2025-12-31"),
    )

    # 添加数据预处理器
    dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
    dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))

    # 收集指数成分过滤器
    filters: dict[str, list[str]] = lab.load_component_filters(index_symbol, start, end)

    # 准备特征和标签数据
    dataset.prepare_data(filters, max_workers=16)

    # 数据预处理
    dataset.process_data()

    # 保存到文件缓存（供后续模型训练使用）
    lab.save_dataset(name, dataset)
    print(f"数据集已保存：{name}")


if __name__ == "__main__":
    main()