"""Visualization module for the Global Liquidity & Productivity Dashboard."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .compute_indices import classify_quadrant

LOGGER = logging.getLogger(__name__)
plt.style.use("seaborn-v0_8")


@dataclass
class FigurePaths:
    outputs_dir: Path

    def path(self, name: str) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        return self.outputs_dir / name


def save_line_chart(series_dict: Dict[str, pd.Series], title: str, figure_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, series in series_dict.items():
        ax.plot(series.index, series.values, label=label)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    LOGGER.info("Saved chart: %s", figure_path)


def plot_creation_flow_lines(creation_df: pd.DataFrame, flow_df: pd.DataFrame, outputs: FigurePaths) -> None:
    save_line_chart({"Creation Index": creation_df["creation_z"]}, "Creation Index", outputs.path("creation_index.png"))
    save_line_chart({"Flow Index": flow_df["flow_index"]}, "Flow Index", outputs.path("flow_index.png"))


def plot_absorption_efficiency(absorption_df: pd.DataFrame, productivity_df: pd.DataFrame, outputs: FigurePaths) -> None:
    save_line_chart({"Absorption Index": absorption_df["absorption_index"]}, "Absorption Index", outputs.path("absorption_index.png"))
    save_line_chart({"Productivity Momentum": productivity_df["productivity_momentum"]}, "Productivity Momentum", outputs.path("productivity_momentum.png"))


def plot_composite(composite_df: pd.DataFrame, outputs: FigurePaths) -> None:
    to_plot = {
        "Creation": composite_df["creation"],
        "Flow": composite_df["flow"],
        "Utilization": composite_df["utilization"],
        "Consumption": composite_df["consumption"],
        "Efficiency": composite_df["efficiency"],
        "Composite": composite_df["composite_index"],
    }
    save_line_chart(to_plot, "Global Liquidity & Productivity Composite", outputs.path("composite_index.png"))


def plot_four_quadrant_panel(creation: pd.Series, absorption: pd.Series, outputs: FigurePaths) -> Path:
    recent_creation = creation.tail(24)
    recent_absorption = absorption.tail(24)
    figure_path = outputs.path("four_quadrant_panel.png")
    if recent_creation.empty or recent_absorption.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "Insufficient data for quadrant panel", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        fig.savefig(figure_path)
        plt.close(fig)
        LOGGER.warning("Skipped four quadrant panel due to insufficient data")
        return figure_path
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)

    ax.fill_between([-3, 3], 0, 3, color="green", alpha=0.1)
    ax.fill_between([-3, 0], 0, 3, color="yellow", alpha=0.1)
    ax.fill_between([-3, 0], -3, 0, color="red", alpha=0.1)
    ax.fill_between([0, 3], -3, 0, color="orange", alpha=0.1)

    points = list(zip(recent_creation.values, recent_absorption.values))
    colors = np.linspace(0.2, 1.0, len(points))
    for idx, ((x, y), color) in enumerate(zip(points, colors)):
        ax.scatter(x, y, color=plt.cm.Blues(color), alpha=0.9)
        ax.annotate(str(recent_creation.index[idx].date()), (x, y), fontsize=6, alpha=0.6)

    latest_point = (recent_creation.iloc[-1], recent_absorption.iloc[-1])
    ax.scatter(*latest_point, color="black", s=80, marker="*", label="Latest")
    regime_text = classify_quadrant(*latest_point)
    ax.set_title(f"Creation vs Absorption — {regime_text}")
    ax.set_xlabel("Creation Z-Score")
    ax.set_ylabel("Absorption Z-Score")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    LOGGER.info("Saved four quadrant panel: %s", figure_path)
    return figure_path


def build_regional_heatmap(etf_prices: pd.DataFrame, windows: Iterable[int], outputs: FigurePaths) -> Tuple[Path, Path]:
    heatmap_data = {}
    monthly_prices = etf_prices.resample("M").last().dropna(how="all")
    for region, series in monthly_prices.items():
        returns = {}
        for window in windows:
            log_ret = np.log(series).diff(window)
            if log_ret.dropna().empty:
                score = np.nan
            else:
                z = (log_ret - log_ret.mean()) / log_ret.std(ddof=0)
                score = z.iloc[-1]
            returns[f"{window}M"] = score
        heatmap_data[region] = returns
    heatmap_df = pd.DataFrame(heatmap_data).T
    heatmap_csv = outputs.path("regional_heatmap.csv")
    heatmap_df.to_csv(heatmap_csv)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap_df.values, cmap="RdYlGn")
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns)
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)
    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            ax.text(j, i, f"{heatmap_df.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    ax.set_title("Regional ETF Heatmap (Log Return Z-Scores)")
    fig.tight_layout()
    heatmap_png = outputs.path("regional_heatmap.png")
    fig.savefig(heatmap_png)
    plt.close(fig)
    LOGGER.info("Saved regional heatmap: %s", heatmap_png)
    return heatmap_csv, heatmap_png


def traffic_light_dashboard(z_scores: Dict[str, float], delta_composite: float, outputs: FigurePaths) -> Path:
    colors = {"green": "#00A676", "yellow": "#FFB400", "red": "#D7263D"}
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    keys = list(z_scores.keys())
    for idx, key in enumerate(keys):
        z = z_scores[key]
        if z > 0.5:
            color = colors["green"]
            label = "expanding"
        elif z < -0.5:
            color = colors["red"]
            label = "contracting"
        else:
            color = colors["yellow"]
            label = "neutral"
        rect = Rectangle((idx * 1.0, 0), 0.9, 0.5, color=color)
        ax.add_patch(rect)
        ax.text(idx * 1.0 + 0.45, 0.25, key.upper(), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(idx * 1.0 + 0.45, -0.1, f"{label}\n(z={z:.2f})", ha="center", va="top")

    if delta_composite > 0.1:
        arrow = "▲"
    elif delta_composite < -0.1:
        arrow = "▼"
    else:
        arrow = "→"
    ax.text(len(keys) + 0.1, 0.25, f"Composite {arrow}", va="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    figure_path = outputs.path("traffic_light_dashboard.png")
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    status_path = outputs.path("traffic_light_status.json")
    with status_path.open("w", encoding="utf-8") as handle:
        json.dump({"z_scores": z_scores, "delta_composite": delta_composite, "arrow": arrow}, handle, indent=2)

    LOGGER.info("Saved traffic light dashboard: %s", figure_path)
    return figure_path


__all__ = [
    "FigurePaths",
    "plot_creation_flow_lines",
    "plot_absorption_efficiency",
    "plot_composite",
    "plot_four_quadrant_panel",
    "build_regional_heatmap",
    "traffic_light_dashboard",
]
