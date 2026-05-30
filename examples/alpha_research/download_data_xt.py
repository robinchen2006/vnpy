# 加载模块
from datetime import datetime
import os

from tqdm import tqdm
from xtquant import xtdata

from vnpy.trader.database import DB_TZ
from vnpy.trader.datafeed import get_datafeed
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import HistoryRequest

from vnpy.alpha import AlphaLab, logger

# 设置下载参数
task_name = "csi300"
index_symbol = "000300.SSE"
xt_index_symbol = "000300.SH"

start_date = "20050101"
end_date = "20251231"

intervals = [
    Interval.DAILY,
]
current_folder = os.path.dirname(os.path.abspath(__file__))
task_path = os.path.join(current_folder, f"./lab/{task_name}")

lab = AlphaLab(task_path)    # 指定数据文件夹

# 初始化数据服务（这里配置使用的迅投研）
datafeed = get_datafeed()
datafeed.init()

# 下载历史成分股信息
# xtdata.download_sector_data()

# xtdata.download_history_data("", "stocklistchange", "", "")


# 查询交易日历
days = xtdata.get_trading_calendar(market="SZ", start_time=start_date, end_time=end_date)

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
    symbol, exchange_str = vt_symbol.split(".")

    for interval in intervals:
        req = HistoryRequest(symbol, Exchange(exchange_str), start, end, interval)
        bars = datafeed.query_bar_history(req)

        if bars:
            lab.save_bar_data(bars)
        else:
            logger(interval, vt_symbol)


# 添加回测参数配置
for vt_symbol in component_symbols:
    lab.add_contract_setting(
        vt_symbol,
        long_rate=5/10000,
        short_rate=10/10000,
        size=1,
        pricetick=0.0001,
    )