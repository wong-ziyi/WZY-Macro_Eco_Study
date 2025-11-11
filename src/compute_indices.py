"""Index construction for the Global Liquidity & Productivity Dashboard."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
except ImportError:  # pragma: no cover - statsmodels optional
    MarkovRegression = None

LOGGER = logging.getLogger(__name__)


def zscore(series: pd.Series) -> pd.Series:
    series = series.copy()
    if series.empty:
        return series
    return (series - series.mean()) / series.std(ddof=0)


def monthly_resample(series: pd.Series, method: str = "last") -> pd.Series:
    if series.empty:
        return series
    if method == "last":
        return series.resample("M").last().dropna()
    if method == "mean":
        return series.resample("M").mean().dropna()
    raise ValueError(f"Unsupported resample method: {method}")


def combine_cb_assets(cb_series: Dict[str, pd.Series]) -> pd.Series:
    if not cb_series:
        return pd.Series(dtype=float)
    aligned = pd.concat(cb_series.values(), axis=1).dropna(how="all")
    aggregate = aligned.sum(axis=1)
    aggregate.name = "global_cb_assets"
    return aggregate


def compute_creation_index(cb_series: Dict[str, pd.Series]) -> pd.DataFrame:
    aggregate = combine_cb_assets(cb_series)
    df = pd.DataFrame({name: monthly_resample(series) for name, series in cb_series.items()})
    df["global_cb_assets"] = monthly_resample(aggregate)
    df["global_liquidity_proxy"] = np.log(df["global_cb_assets"]).diff().fillna(0)
    df["creation_z"] = zscore(df["global_liquidity_proxy"]).clip(-3, 3)
    return df


def compute_flow_index(flow_inputs: Dict[str, pd.Series], equity_df: pd.DataFrame,
                        bond_df: pd.DataFrame, stablecoin: pd.Series) -> pd.DataFrame:
    fx = monthly_resample(flow_inputs.get("fx", pd.Series(dtype=float)))
    rates = monthly_resample(flow_inputs.get("rates", pd.Series(dtype=float)))
    credit = monthly_resample(flow_inputs.get("credit", pd.Series(dtype=float)))

    equities = equity_df.resample("M").last().pct_change().mean(axis=1)
    bonds = bond_df.resample("M").last().pct_change().mean(axis=1)
    commodities = flow_inputs.get("commodities")
    commodity_ret = (
        commodities.resample("M").last().pct_change().mean(axis=1)
        if isinstance(commodities, pd.DataFrame) and not commodities.empty else pd.Series(dtype=float)
    )
    stablecoin_growth = stablecoin.resample("M").last().pct_change()

    df = pd.DataFrame({
        "z_fx": -zscore(fx.pct_change()),
        "z_rates": -zscore(rates.pct_change()),
        "z_credit": zscore(-credit.pct_change()),
        "z_equities": zscore(equities),
        "z_bonds": zscore(bonds),
        "z_commodities": zscore(commodity_ret),
        "z_stablecoin": zscore(stablecoin_growth),
    })
    if df.empty:
        df = pd.DataFrame(columns=[
            "z_fx",
            "z_rates",
            "z_credit",
            "z_equities",
            "z_bonds",
            "z_commodities",
            "z_stablecoin",
        ])
    else:
        df = df.dropna(how="all")

    df["flow_index"] = (
        0.25 * df.get("z_fx", 0)
        + 0.25 * df.get("z_rates", 0)
        + 0.20 * df.get("z_credit", 0)
        + 0.20 * df.get("z_equities", 0)
        + 0.10 * df.get("z_bonds", 0)
        + 0.05 * df.get("z_stablecoin", 0)
    )
    df["flow_index"] = df["flow_index"].clip(-3, 3)
    return df


def compute_absorption_indices(utilization: pd.Series, consumption: pd.Series) -> pd.DataFrame:
    util_z = zscore(utilization)
    cons_z = zscore(consumption)
    df = pd.DataFrame({
        "utilization_z": util_z,
        "consumption_z": cons_z,
    })
    if df.empty:
        df = pd.DataFrame(columns=["utilization_z", "consumption_z"])
    df["absorption_index"] = df[["utilization_z", "consumption_z"]].mean(axis=1)
    return df


def compute_productivity_index(tfp: pd.Series, rd: pd.Series, ai_proxy: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({
        "tfp_z": zscore(tfp),
        "rd_z": zscore(rd),
        "ai_z": zscore(ai_proxy),
    })
    if df.empty:
        df = pd.DataFrame(columns=["tfp_z", "rd_z", "ai_z"])
    else:
        df = df.dropna(how="all")
    df["productivity_momentum"] = df.mean(axis=1)
    return df


def combine_components(creation: pd.Series, flow: pd.Series, utilization: pd.Series,
                       consumption: pd.Series, efficiency: pd.Series, weights: Dict[str, float]) -> pd.DataFrame:
    df = pd.concat([
        creation.rename("creation"),
        flow.rename("flow"),
        utilization.rename("utilization"),
        consumption.rename("consumption"),
        efficiency.rename("efficiency"),
    ], axis=1)
    df = df.fillna(method="ffill").fillna(0)
    df["composite_index"] = (
        weights.get("creation", 0.3) * df["creation"]
        + weights.get("flow", 0.3) * df["flow"]
        + weights.get("utilization", 0.2) * df["utilization"]
        + weights.get("consumption", 0.1) * df["consumption"]
        + weights.get("efficiency", 0.1) * df["efficiency"]
    )
    df["composite_index"] = df["composite_index"].clip(-3, 3)
    return df


@dataclass
class RollingPCAResult:
    loadings: pd.DataFrame
    scores: pd.Series


def compute_rolling_pca(data: pd.DataFrame, window: int = 36) -> RollingPCAResult:
    if data.empty:
        return RollingPCAResult(loadings=pd.DataFrame(), scores=pd.Series(dtype=float))
    loadings = {}
    scores = []
    for end in range(window, len(data) + 1):
        window_df = data.iloc[end - window:end]
        if window_df.isna().any().any():
            window_df = window_df.dropna()
        if len(window_df) < 3:
            continue
        pca = PCA(n_components=1)
        comp = pca.fit_transform(window_df)
        loadings[data.index[end - 1]] = pca.components_[0]
        scores.append((data.index[end - 1], comp[-1, 0]))
    loadings_df = pd.DataFrame(loadings, index=data.columns).T
    scores_series = pd.Series(dict(scores)).rename("liquidity_factor")
    return RollingPCAResult(loadings=loadings_df, scores=scores_series)


def detect_regimes(series: pd.Series, k_regimes: int = 4) -> pd.DataFrame:
    if series.empty or MarkovRegression is None:
        LOGGER.warning("Regime detection skipped due to missing data or statsmodels")
        return pd.DataFrame()
    try:
        model = MarkovRegression(series.dropna(), k_regimes=k_regimes, trend="c", switching_variance=True)
        fit = model.fit(disp=False)
        regimes = fit.smoothed_marginal_probabilities.idxmax(axis=1)
        probabilities = fit.smoothed_marginal_probabilities
        result = pd.concat([regimes.rename("regime"), probabilities], axis=1)
        return result
    except Exception as exc:  # pragma: no cover - statsmodels can fail to converge
        LOGGER.warning("Regime detection failed: %s", exc)
        return pd.DataFrame()


SCENARIO_LABELS = {
    0: "RISK-ON EXPANSION",
    1: "LATE-CYCLE REFLATION",
    2: "RISK-OFF CONTRACTION",
    3: "LIQUIDITY TRAP",
}


def classify_quadrant(creation_z: float, absorption_z: float) -> str:
    if creation_z >= 0 and absorption_z >= 0:
        return SCENARIO_LABELS[0]
    if creation_z < 0 and absorption_z >= 0:
        return SCENARIO_LABELS[1]
    if creation_z < 0 and absorption_z < 0:
        return SCENARIO_LABELS[2]
    return SCENARIO_LABELS[3]


def scenario_tree(base_levels: Dict[str, float], shocks: Dict[str, float], weights: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame({"base": base_levels})
    df["shock"] = df.index.map(shocks.get)
    df["post_shock"] = df["base"] + df["shock"].fillna(0)
    df.loc["composite", "base"] = sum(weights.get(key, 0) * base_levels.get(key, np.nan) for key in weights)
    df.loc["composite", "shock"] = sum(weights.get(key, 0) * shocks.get(key, 0) for key in weights)
    df.loc["composite", "post_shock"] = df.loc["composite", "base"] + df.loc["composite", "shock"]
    return df


__all__ = [
    "zscore",
    "monthly_resample",
    "combine_cb_assets",
    "compute_creation_index",
    "compute_flow_index",
    "compute_absorption_indices",
    "compute_productivity_index",
    "combine_components",
    "compute_rolling_pca",
    "RollingPCAResult",
    "detect_regimes",
    "classify_quadrant",
    "scenario_tree",
]
