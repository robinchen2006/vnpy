# 准备数据

# 过滤Alphalens的warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 加载模块
import polars as pl

from vnpy.trader.constant import Interval

from vnpy.alpha import AlphaLab

# 创建数据中心
lab: AlphaLab = AlphaLab("./lab/csi300")

# 设置任务参数
name = "300_mlp"
index_symbol: str = "000300.SSE"
start: str = "2008-01-01"
end: str = "2023-12-31"
interval: Interval = Interval.DAILY
extended_days: int = 100

# 加载所有成分股代码
component_symbols: list[str] = lab.load_component_symbols(index_symbol, start, end)
print(f"成分股代码加载完成，共 {len(component_symbols)} 只")
