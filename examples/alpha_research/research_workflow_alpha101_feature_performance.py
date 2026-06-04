"""Feature performance analysis script.

Converted from research_workflow_alpha101_feature_performance.ipynb.
Runs dataset.show_feature_performance, saves each figure as a JPG,
and saves analysis tables to CSV.
"""

from __future__ import annotations

import io
import os
import unittest.mock as mock
import warnings
from contextlib import redirect_stdout
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from alphalens import performance as perf  # type: ignore
from alphalens import utils as alphalens_utils  # type: ignore
from alphalens.utils import get_clean_factor_and_forward_returns  # type: ignore
from scipy import stats

from vnpy.alpha import AlphaLab

warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------------------------------------------------
def _save_figures_as_jpg(output_dir: str, file_prefix: str) -> list[str]:
    """Save each open matplotlib figure as an individual JPG file."""
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []
    idx = 1
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        # 跳过没有任何实际内容的空白 figure
        has_content = any(
            len(ax.lines) > 0
            or len(ax.collections) > 0
            or len(ax.patches) > 0
            or len(ax.images) > 0
            or len(ax.tables) > 0
            or ax.get_title() != ""
            for ax in fig.get_axes()
        )
        if not has_content:
            continue
        path = os.path.join(output_dir, f"{file_prefix}_{idx:02d}.jpg")
        fig.savefig(path, dpi=200, bbox_inches="tight", format="jpg")
        paths.append(path)
        idx += 1
    return paths


# -----------------------------------------------------------------------------
def _build_clean_data(dataset, feature_name: str) -> pd.DataFrame:
    """Build Alphalens clean_data with the same logic as show_feature_performance."""
    starts = [pd.Timestamp(p[0]) for p in dataset.data_periods.values()]
    ends = [pd.Timestamp(p[1]) for p in dataset.data_periods.values()]
    start = min(starts).to_pydatetime()
    end = max(ends).to_pydatetime()

    result_df: pl.DataFrame = dataset.result_df.filter(
        (pl.col("datetime") >= pl.lit(start)) & (pl.col("datetime") <= pl.lit(end))
    )
    learn_df: pl.DataFrame = dataset.learn_df.filter(
        (pl.col("datetime") >= pl.lit(start)) & (pl.col("datetime") <= pl.lit(end))
    )
    merged = (
        result_df.select(["datetime", "vt_symbol", "close"])
        .join(
            learn_df.select(["datetime", "vt_symbol", feature_name]),
            on=["datetime", "vt_symbol"],
            how="inner",
        )
        .fill_nan(None)
        .drop_nulls()
    )
    feature_s = (
        merged.select(["datetime", "vt_symbol", feature_name])
        .to_pandas()
        .set_index(["datetime", "vt_symbol"])[feature_name]
    )
    price_df = (
        merged.select(["datetime", "vt_symbol", "close"])
        .to_pandas()
        .pivot(index="datetime", columns="vt_symbol", values="close")
    )
    return get_clean_factor_and_forward_returns(feature_s, price_df, quantiles=10)


# -----------------------------------------------------------------------------
def _save_csv(
    dataset, feature_name: str, output_dir: str, file_prefix: str
) -> list[str]:
    """Save analysis tables shown in console output to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    clean_data = _build_clean_data(dataset, feature_name)
    paths: list[str] = []

    # quantiles statistics (min, max, mean, std, count, count %)
    qs = clean_data.groupby("factor_quantile")["factor"].agg(
        ["min", "max", "mean", "std", "count"]
    )
    qs["count %"] = qs["count"] / qs["count"].sum() * 100.0
    p = os.path.join(output_dir, f"{file_prefix}_quantiles_statistics.csv")
    qs.to_csv(p, encoding="utf-8-sig")
    paths.append(p)

    # returns analysis (same logic as create_returns_tear_sheet)
    factor_returns = perf.factor_returns(clean_data, demeaned=True, group_adjust=False)
    mean_quant_ret, _ = perf.mean_return_by_quantile(
        clean_data,
        by_group=False,
        demeaned=True,
        group_adjust=False,
    )
    mean_quant_rateret = mean_quant_ret.apply(
        alphalens_utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret.columns[0],
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        clean_data,
        by_date=True,
        by_group=False,
        demeaned=True,
        group_adjust=False,
    )
    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        alphalens_utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )
    compstd_quant_daily = std_quant_daily.apply(
        alphalens_utils.std_conversion,
        axis=0,
        base_period=std_quant_daily.columns[0],
    )
    alpha_beta = perf.factor_alpha_beta(
        clean_data,
        factor_returns,
        demeaned=True,
        group_adjust=False,
    )
    mean_ret_spread_quant, _ = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        clean_data["factor_quantile"].max(),
        clean_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )
    returns_table = pd.DataFrame()
    returns_table = pd.concat([returns_table, alpha_beta])
    returns_table.loc["Mean Period Wise Return Top Quantile (bps)"] = (
        mean_quant_rateret.iloc[-1] * 10000
    )
    returns_table.loc["Mean Period Wise Return Bottom Quantile (bps)"] = (
        mean_quant_rateret.iloc[0] * 10000
    )
    returns_table.loc["Mean Period Wise Spread (bps)"] = (
        mean_ret_spread_quant.mean() * 10000
    )
    p = os.path.join(output_dir, f"{file_prefix}_returns_analysis.csv")
    returns_table.apply(lambda x: x.round(3)).to_csv(p, encoding="utf-8-sig")
    paths.append(p)

    # information analysis (same logic as create_information_tear_sheet)
    ic = perf.factor_information_coefficient(clean_data, False)
    information_table = pd.DataFrame()
    information_table["IC Mean"] = ic.mean()
    information_table["IC Std."] = ic.std()
    information_table["Risk-Adjusted IC"] = ic.mean() / ic.std()
    ttest_result = cast(Any, stats.ttest_1samp(ic, 0))
    information_table["t-stat(IC)"] = pd.Series(
        ttest_result.statistic, index=ic.columns
    )
    information_table["p-value(IC)"] = pd.Series(ttest_result.pvalue, index=ic.columns)
    information_table["IC Skew"] = stats.skew(ic)
    information_table["IC Kurtosis"] = stats.kurtosis(ic)
    p = os.path.join(output_dir, f"{file_prefix}_information_analysis.csv")
    information_table.apply(lambda x: x.round(3)).T.to_csv(p, encoding="utf-8-sig")
    paths.append(p)

    # turnover analysis (same logic as create_turnover_tear_sheet)
    input_periods = alphalens_utils.get_forward_returns_columns(
        clean_data.columns,
        require_exact_day_multiple=True,
    ).to_numpy()
    turnover_periods = alphalens_utils.timedelta_strings_to_integers(input_periods)
    quantile_factor = clean_data["factor_quantile"]
    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods
    }
    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(clean_data, period)
            for period in turnover_periods
        ],
        axis=1,
    )
    turnover_table = pd.DataFrame()
    for period in sorted(quantile_turnover.keys()):
        for quantile, p_data in quantile_turnover[period].items():
            turnover_table.loc[
                f"Quantile {quantile} Mean Turnover ",
                f"{period}D",
            ] = p_data.mean()

    auto_corr = pd.DataFrame()
    for period, p_data in autocorrelation.items():
        auto_corr.loc[
            "Mean Factor Rank Autocorrelation",
            f"{period}D",
        ] = p_data.mean()

    p = os.path.join(output_dir, f"{file_prefix}_turnover_analysis.csv")
    turnover_table.apply(lambda x: x.round(3)).to_csv(p, encoding="utf-8-sig")
    paths.append(p)
    p = os.path.join(output_dir, f"{file_prefix}_turnover_autocorrelation.csv")
    auto_corr.apply(lambda x: x.round(3)).to_csv(p, encoding="utf-8-sig")
    paths.append(p)

    return paths


# -----------------------------------------------------------------------------
def main() -> None:
    name = "300_lgb"
    feature_name = "alpha2"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csi300_root = os.path.join(current_dir, "lab", "csi300")
    dataset = AlphaLab(csi300_root).load_dataset(name)

    if dataset is None:
        raise FileNotFoundError(f"Dataset '{name}' not found under {csi300_root}")
    # Patch plt.show 为空操作，防止 show_feature_performance 内部调用
    # plt.show() 时清空所有 figure

    with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.close"):
        buf = io.StringIO()
        with redirect_stdout(buf):
            dataset.show_feature_performance(feature_name)

    print(buf.getvalue(), end="")

    output_dir = os.path.join(csi300_root, "reports", name, feature_name)
    prefix = f"{feature_name}"

    # figure 此时还存活，立即保存
    jpg_files = _save_figures_as_jpg(output_dir, prefix)

    # 手动清理
    plt.close("all")

    csv_files = _save_csv(dataset, feature_name, output_dir, prefix)

    print(f"\n图片已保存 ({len(jpg_files)} 张):")
    for p in jpg_files:
        print(f"  {p}")
    print("CSV已保存:")
    for p in csv_files:
        print(f"  {p}")


if __name__ == "__main__":
    main()
