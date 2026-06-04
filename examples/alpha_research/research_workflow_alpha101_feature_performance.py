"""Feature performance analysis script.

Converted from research_workflow_alpha101_feature_performance.ipynb.
Refactored into reusable modules and supports batch analysis for all alpha factors.
"""

from __future__ import annotations

import io
import os
import unittest.mock as mock
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime configuration for feature analysis."""

    dataset_name: str = "300_lgb"
    alpha_prefix: str = "alpha"


class DatasetDataView:
    """Cached dataset views to avoid repeating expensive filtering work."""

    def __init__(self, dataset) -> None:
        starts = [pd.Timestamp(period[0]) for period in dataset.data_periods.values()]
        ends = [pd.Timestamp(period[1]) for period in dataset.data_periods.values()]

        self.start = min(starts).to_pydatetime()
        self.end = max(ends).to_pydatetime()

        self.result_df: pl.DataFrame = dataset.result_df.filter(
            (pl.col("datetime") >= pl.lit(self.start))
            & (pl.col("datetime") <= pl.lit(self.end))
        )
        self.learn_df: pl.DataFrame = dataset.learn_df.filter(
            (pl.col("datetime") >= pl.lit(self.start))
            & (pl.col("datetime") <= pl.lit(self.end))
        )
        self.base_price_df: pl.DataFrame = self.result_df.select(
            ["datetime", "vt_symbol", "close"]
        )


def _save_figures_as_jpg(output_dir: Path, file_prefix: str) -> list[Path]:
    """Save each open matplotlib figure as an individual JPG file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    idx = 1

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
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

        path = output_dir / f"{file_prefix}_{idx:02d}.jpg"
        fig.savefig(path, dpi=200, bbox_inches="tight", format="jpg")
        paths.append(path)
        idx += 1

    return paths


def _build_clean_data(view: DatasetDataView, feature_name: str) -> pd.DataFrame:
    """Build Alphalens clean_data with the same logic as show_feature_performance."""
    merged = (
        view.base_price_df.join(
            view.learn_df.select(["datetime", "vt_symbol", feature_name]),
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


def _save_csv(
    view: DatasetDataView,
    feature_name: str,
    output_dir: Path,
    file_prefix: str,
) -> list[Path]:
    """Save analysis tables shown in console output to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_data = _build_clean_data(view, feature_name)
    paths: list[Path] = []

    qs = clean_data.groupby("factor_quantile")["factor"].agg(
        ["min", "max", "mean", "std", "count"]
    )
    qs["count %"] = qs["count"] / qs["count"].sum() * 100.0
    p = output_dir / f"{file_prefix}_quantiles_statistics.csv"
    qs.to_csv(p, encoding="utf-8-sig")
    paths.append(p)

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
    p = output_dir / f"{file_prefix}_returns_analysis.csv"
    returns_table.apply(lambda x: x.round(3)).to_csv(p, encoding="utf-8-sig")
    paths.append(p)

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
    p = output_dir / f"{file_prefix}_information_analysis.csv"
    information_table.apply(lambda x: x.round(3)).T.to_csv(p, encoding="utf-8-sig")
    paths.append(p)

    input_periods = alphalens_utils.get_forward_returns_columns(
        clean_data.columns,
        require_exact_day_multiple=True,
    ).to_numpy()
    turnover_periods = alphalens_utils.timedelta_strings_to_integers(input_periods)
    quantile_factor = clean_data["factor_quantile"]
    quantile_turnover = {
        period: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, quantile, period)
                for quantile in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for period in turnover_periods
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
        for quantile, period_data in quantile_turnover[period].items():
            turnover_table.loc[
                f"Quantile {quantile} Mean Turnover ",
                f"{period}D",
            ] = period_data.mean()

    auto_corr = pd.DataFrame()
    for period, period_data in autocorrelation.items():
        auto_corr.loc[
            "Mean Factor Rank Autocorrelation",
            f"{period}D",
        ] = period_data.mean()

    p = output_dir / f"{file_prefix}_turnover_analysis.csv"
    turnover_table.apply(lambda x: x.round(3)).to_csv(p, encoding="utf-8-sig")
    paths.append(p)

    p = output_dir / f"{file_prefix}_turnover_autocorrelation.csv"
    auto_corr.apply(lambda x: x.round(3)).to_csv(p, encoding="utf-8-sig")
    paths.append(p)

    return paths


def _list_alpha_features(dataset, alpha_prefix: str = "alpha") -> list[str]:
    """Collect all alpha factor names from learn_df columns."""
    features = [
        col
        for col in dataset.learn_df.columns
        if col.startswith(alpha_prefix) and col not in {"datetime", "vt_symbol"}
    ]
    return sorted(features)


def _save_initial_alpha_output(
    view: DatasetDataView,
    output_dir: Path,
    feature_names: list[str],
) -> Path:
    """Save the earliest timestamp snapshot for all alpha factors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    first_dt = view.learn_df.select(pl.col("datetime").min()).item()
    initial_df = view.learn_df.filter(pl.col("datetime") == first_dt).select(
        ["datetime", "vt_symbol", *feature_names]
    )

    output_path = output_dir / "all_alpha_initial_output.csv"
    initial_df.to_pandas().to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def _run_single_feature(
    dataset,
    view: DatasetDataView,
    feature_name: str,
    report_root: Path,
) -> tuple[list[Path], list[Path]]:
    """Run figure and table output for one feature."""
    output_dir = report_root / feature_name

    plt.close("all")
    with mock.patch("matplotlib.pyplot.show"), mock.patch("matplotlib.pyplot.close"):
        buf = io.StringIO()
        with redirect_stdout(buf):
            dataset.show_feature_performance(feature_name)
    print(buf.getvalue(), end="")

    jpg_files = _save_figures_as_jpg(output_dir, feature_name)
    plt.close("all")
    csv_files = _save_csv(view, feature_name, output_dir, feature_name)
    return jpg_files, csv_files


def main() -> None:
    config = AnalysisConfig()

    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    csi300_root = current_dir / "lab" / "csi300"
    dataset = AlphaLab(str(csi300_root)).load_dataset(config.dataset_name)

    if dataset is None:
        raise FileNotFoundError(
            f"Dataset '{config.dataset_name}' not found under {csi300_root}"
        )

    view = DatasetDataView(dataset)
    feature_names = _list_alpha_features(dataset, config.alpha_prefix)
    if not feature_names:
        raise ValueError(
            f"No alpha features with prefix '{config.alpha_prefix}' were found"
        )

    report_root = csi300_root / "reports" / config.dataset_name
    initial_output_path = _save_initial_alpha_output(view, report_root, feature_names)

    print(f"共识别到 {len(feature_names)} 个alpha因子")
    print(f"初始输出已保存: {initial_output_path}")

    all_jpg_files: list[Path] = []
    all_csv_files: list[Path] = []
    failed_features: list[tuple[str, str]] = []

    for idx, feature_name in enumerate(feature_names, start=1):
        print(f"\n[{idx}/{len(feature_names)}] 开始分析 {feature_name}")
        try:
            jpg_files, csv_files = _run_single_feature(
                dataset,
                view,
                feature_name,
                report_root,
            )
            all_jpg_files.extend(jpg_files)
            all_csv_files.extend(csv_files)
            print(
                f"[{idx}/{len(feature_names)}] 完成 {feature_name} "
                f"(图片 {len(jpg_files)} 张, CSV {len(csv_files)} 个)"
            )
        except Exception as exc:  # noqa: BLE001
            failed_features.append((feature_name, str(exc)))
            print(f"[{idx}/{len(feature_names)}] 失败 {feature_name}: {exc}")

    print("\n汇总:")
    print(f"  Alpha因子总数: {len(feature_names)}")
    print(f"  图片总数: {len(all_jpg_files)}")
    print(f"  CSV总数: {len(all_csv_files)}")
    print(f"  失败数: {len(failed_features)}")

    if failed_features:
        print("失败明细:")
        for feature_name, err in failed_features:
            print(f"  {feature_name}: {err}")


if __name__ == "__main__":
    main()
