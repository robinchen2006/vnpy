# 加载模块
import os
from datetime import datetime

from tqdm import tqdm
from vnpy_dolphindb.database import DolphindbDatabase
from xtquant import xtdata

from vnpy.alpha import AlphaLab, logger
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ

# 设置下载参数
task_name = "csi300_2"
index_symbol = "000300.SSE"
xt_index_symbol = "000300.SH"

start_date = "20050101"
end_date = "20251231"

intervals = [
    Interval.DAILY,
]
current_folder = os.path.dirname(os.path.abspath(__file__))
task_path = os.path.join(current_folder, f"./lab/{task_name}")

lab = AlphaLab(task_path)  # 指定数据文件夹

dbm = DolphindbDatabase()
# 下载历史成分股信息
# xtdata.download_sector_data()

# xtdata.download_history_data("", "stocklistchange", "", "")


# 查询交易日历
days = xtdata.get_trading_calendar(
    market="SZ", start_time=start_date, end_time=end_date
)

# 轮询获取指数成本股
index_components = {}
end_datetime = datetime.strptime(end_date, "%Y%m%d")
for i in days:
    dt = datetime.strptime(i, "%Y%m%d")
    if dt > end_datetime:
        continue

    xt_symbols = xtdata.get_stock_list_in_sector(xt_index_symbol, i)

    vt_symbols: list = []
    for xt_symbol in xt_symbols:
        vt_symbol = xt_symbol.replace("SH", "SSE").replace("SZ", "SZSE")
        vt_symbols.append(vt_symbol)

    index_components[dt.strftime("%Y-%m-%d")] = vt_symbols

# 保存到数据中心
lab.save_component_data(index_symbol, index_components)

component_symbols = lab.load_component_symbols(index_symbol, start_date, end_date)

# 转换时间格式
start = datetime.strptime(start_date, "%Y%m%d")
start.replace(tzinfo=DB_TZ)

end = datetime.strptime(end_date, "%Y%m%d")
end.replace(tzinfo=DB_TZ)

# 除了成分股，还要下载指数数据
task_symbols = component_symbols + [index_symbol]

# 遍历下载数据
for vt_symbol in tqdm(task_symbols):
    if "000001" in vt_symbol:
        logger.info(f"Skipping index symbol {vt_symbol} for bar data download.")
    symbol, exchange_str = vt_symbol.split(".")

    for interval in intervals:
        bars = dbm.load_bar_data_ext(
            symbol, Exchange(exchange_str), interval, start, end, "backward"
        )

        if bars:
            lab.save_bar_data(bars)
            logger.info(
                f"Downloaded data for {vt_symbol} with interval {interval}, total bars: {len(bars)}"
            )
        else:
            logger.warning(
                f"Failed to download data for {vt_symbol} with interval {interval}"
            )


# 添加回测参数配置
for vt_symbol in component_symbols:
    lab.add_contract_setting(
        vt_symbol,
        long_rate=5 / 10000,
        short_rate=10 / 10000,
        size=1,
        pricetick=0.0001,
    )
