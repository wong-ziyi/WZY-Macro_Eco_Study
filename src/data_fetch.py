"""Data acquisition utilities for the Global Liquidity & Productivity Dashboard."""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests
import yfinance as yf

try:  # pandas-datareader is optional but recommended for macro series
    from pandas_datareader import data as pdr
    from pandas_datareader import wb
except ImportError:  # pragma: no cover - handled gracefully at runtime
    pdr = None
    wb = None


LOGGER = logging.getLogger(__name__)
DEFAULT_LOG_LEVEL = logging.INFO


@dataclass
class FetchConfig:
    """Configuration container loaded from ``config/config.json``."""

    raw: Dict
    outputs_dir: Path
    lookback_months: int
    pca_window: int
    regime_states: int
    scenario_shocks: Dict[str, float]

    @classmethod
    def load(cls, config_path: Path) -> "FetchConfig":
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        outputs_dir = Path(payload.get("outputs", "outputs"))
        outputs_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            raw=payload,
            outputs_dir=outputs_dir,
            lookback_months=int(payload.get("lookback_months", 120)),
            pca_window=int(payload.get("pca_window", 36)),
            regime_states=int(payload.get("regime_states", 4)),
            scenario_shocks=payload.get("scenario_shocks", {}),
        )

    def section(self, name: str) -> Dict:
        return self.raw.get(name, {})


@dataclass
class DataFetcher:
    """Wraps all external data retrieval with safe fallbacks and logging."""

    config: FetchConfig
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.session.headers.update({"User-Agent": "GlobalLiquidityDashboard/1.0"})

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_datetime(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df.index.name != "date":
            df.index.name = "date"
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        return (series - series.mean()) / series.std(ddof=0)

    # ------------------------------------------------------------------
    # CREATION LAYER FETCHERS
    # ------------------------------------------------------------------
    def fetch_fed_balance_sheet(self) -> pd.Series:
        ticker = self.config.section("creation").get("fed_ticker", "WALCL")
        LOGGER.info("Fetching Fed balance sheet via yfinance: %s", ticker)
        try:
            data = yf.download(ticker, period="max", interval="1wk", auto_adjust=False, progress=False)
            series = data["Adj Close"].dropna().rename("fed_balance_sheet")
            series.index = pd.to_datetime(series.index)
            return series
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("Fed balance sheet fetch failed: %s", exc)
            return self._fallback_series("FED_WALCL")

    def fetch_tga_balance(self) -> pd.Series:
        ticker = self.config.section("creation").get("tga_ticker", "WTREGEN")
        LOGGER.info("Fetching TGA balance from FRED via yfinance: %s", ticker)
        try:
            data = yf.download(ticker, period="max", interval="1wk", auto_adjust=False, progress=False)
            series = data["Adj Close"].dropna().rename("tga_balance")
            series.index = pd.to_datetime(series.index)
            return series
        except Exception as exc:
            LOGGER.warning("TGA fetch failed: %s", exc)
            return self._fallback_series("TGA_BALANCE")

    def fetch_ust_issuance(self) -> pd.Series:
        ticker = self.config.section("creation").get("ust_issuance_ticker", "USTBOND")
        LOGGER.info("Fetching UST issuance proxy via yfinance: %s", ticker)
        try:
            data = yf.download(ticker, period="max", auto_adjust=False, progress=False)
            series = data["Adj Close"].dropna().rename("ust_issuance")
            series.index = pd.to_datetime(series.index)
            return series
        except Exception as exc:
            LOGGER.warning("UST issuance fetch failed: %s", exc)
            return self._fallback_series("UST_ISSUANCE")

    def fetch_ecb_total_assets(self) -> pd.Series:
        series_code = self.config.section("creation").get("ecb_series")
        if not series_code:
            return self._fallback_series("ECB_ASSETS")
        url = f"https://sdw-wsrest.ecb.europa.eu/service/data/{series_code}?lastNObservations=520"
        LOGGER.info("Fetching ECB assets from SDW: %s", series_code)
        try:
            response = self.session.get(url, headers={"Accept": "application/json"}, timeout=30)
            response.raise_for_status()
            payload = response.json()
            observations = payload["dataSets"][0]["series"]["0:0:0:0:0:0:0:0"].get("observations", {})
            dates = payload["structure"]["dimensions"]["observation"][0]["values"]
            records = []
            for obs_index, (key, value) in enumerate(observations.items()):
                obs_date = dates[int(key)]["id"]
                records.append((pd.Period(obs_date, freq="M").to_timestamp("M"), value[0]))
            if not records:
                raise ValueError("No ECB observations returned")
            series = pd.Series(dict(records), name="ecb_assets").sort_index()
            return series
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("ECB assets fetch failed: %s", exc)
            return self._fallback_series("ECB_ASSETS")

    def fetch_boj_total_assets(self) -> pd.Series:
        xls_url = self.config.section("creation").get("boj_xls_url")
        if not xls_url:
            return self._fallback_series("BOJ_ASSETS")
        LOGGER.info("Fetching BoJ assets from XLS: %s", xls_url)
        try:
            response = self.session.get(xls_url, timeout=30)
            response.raise_for_status()
            series = pd.read_excel(io.BytesIO(response.content), index_col=0).iloc[:, 0]
            series.index = pd.to_datetime(series.index)
            series.name = "boj_assets"
            return series
        except Exception as exc:
            LOGGER.warning("BoJ assets fetch failed: %s", exc)
            return self._fallback_series("BOJ_ASSETS")

    def fetch_pboc_m2(self) -> pd.Series:
        csv_path = Path(self.config.section("creation").get("pboc_local_csv", ""))
        if csv_path.exists():
            LOGGER.info("Loading PBOC M2 from local CSV: %s", csv_path)
            df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            return df.squeeze("columns").rename("pboc_m2")
        LOGGER.warning("Local PBOC M2 CSV missing; falling back to sample data")
        return self._fallback_series("PBOC_M2")

    def fetch_imf_broad_money(self) -> pd.Series:
        LOGGER.info("Fetching IMF broad money via SDMX JSON")
        countries: Iterable[str] = self.config.section("creation").get("imf_country_codes", [])
        try:
            series_frames: List[pd.Series] = []
            for country in countries:
                url = (
                    "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/"
                    f"M.{country}.MABMM01_XDC?startPeriod=2000"
                )
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                payload = response.json()
                observations = (
                    payload.get("CompactData", {})
                    .get("DataSet", {})
                    .get("Series", {})
                    .get("Obs", [])
                )
                if not observations:
                    continue
                data_map = {
                    pd.Period(obs["@TIME_PERIOD"], freq="M").to_timestamp("M"): float(obs["@OBS_VALUE"])
                    for obs in observations
                }
                series = pd.Series(data_map)
                series_frames.append(series.rename(country))
            if not series_frames:
                raise ValueError("No IMF data retrieved")
            combined = pd.concat(series_frames, axis=1).sum(axis=1)
            combined.name = "imf_broad_money"
            return combined.sort_index()
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("IMF broad money fetch failed: %s", exc)
            return self._fallback_series("IMF_WORLD")

    # ------------------------------------------------------------------
    # FLOW LAYER FETCHERS
    # ------------------------------------------------------------------
    def fetch_market_series(self, tickers: Iterable[str], column_name: str) -> pd.DataFrame:
        tickers = list(tickers)
        if not tickers:
            return pd.DataFrame()
        LOGGER.info("Fetching market data via yfinance for %d tickers", len(tickers))
        try:
            data = yf.download(tickers, period="max", auto_adjust=True, progress=False)["Adj Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame(name=tickers[0])
            data = data.loc[:, ~data.columns.duplicated()].dropna(how="all")
            data.columns = pd.Index([f"{column_name}_{col}" for col in data.columns], name="field")
            data.index = pd.to_datetime(data.index)
            return data
        except Exception as exc:
            LOGGER.warning("Market data fetch failed (%s); returning empty frame", exc)
            return pd.DataFrame()

    def fetch_stablecoin_market_cap(self) -> pd.Series:
        endpoint = self.config.section("flow").get("stablecoin_endpoint")
        if not endpoint:
            return pd.Series(dtype=float)
        LOGGER.info("Fetching stablecoin market cap from DefiLlama")
        try:
            response = self.session.get(endpoint, timeout=30)
            response.raise_for_status()
            payload = response.json().get("totalCharts", [])
            data = {datetime.utcfromtimestamp(point[0] / 1000): point[1] for point in payload}
            series = pd.Series(data).sort_index()
            series.name = "stablecoin_market_cap"
            return series
        except Exception as exc:
            LOGGER.warning("Stablecoin market cap fetch failed: %s", exc)
            return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # UTILIZATION & CONSUMPTION FETCHERS
    # ------------------------------------------------------------------
    def fetch_world_bank_indicator(self, indicator: str) -> pd.Series:
        if wb is None:
            LOGGER.warning("pandas-datareader not installed; cannot fetch World Bank indicator %s", indicator)
            return pd.Series(dtype=float)
        LOGGER.info("Fetching World Bank indicator: %s", indicator)
        try:
            df = wb.download(indicator=indicator, country="all", start=1990, end=datetime.utcnow().year)
            df = df[df[indicator].notna()].groupby("year")[indicator].mean()
            df.index = pd.to_datetime(df.index.astype(str))
            return df.rename(indicator)
        except Exception as exc:
            LOGGER.warning("World Bank fetch failed: %s", exc)
            return pd.Series(dtype=float)

    def fetch_oecd_series(self, series: str) -> pd.Series:
        LOGGER.info("Fetching OECD series: %s", series)
        try:
            if pdr is None:
                raise RuntimeError("pandas-datareader is required for OECD series")
            df = pdr.DataReader(series, "oecd")
            df = df.squeeze("columns").dropna()
            df.index = pd.to_datetime(df.index)
            return df.rename(series)
        except Exception as exc:
            LOGGER.warning("OECD fetch failed: %s", exc)
            return pd.Series(dtype=float)

    def fetch_retail_sales(self, tickers: Dict[str, str]) -> pd.DataFrame:
        frames = []
        for region, ticker in tickers.items():
            LOGGER.info("Fetching retail sales series %s (%s)", ticker, region)
            try:
                data = yf.download(ticker, period="max", auto_adjust=False, progress=False)["Adj Close"].dropna()
                data.index = pd.to_datetime(data.index)
                frames.append(data.rename(region))
            except Exception as exc:
                LOGGER.warning("Retail sales fetch failed for %s: %s", ticker, exc)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)

    # ------------------------------------------------------------------
    # FALLBACK UTILITIES
    # ------------------------------------------------------------------
    def _fallback_series(self, series_code: str) -> pd.Series:
        sample_path = Path("data/cb_assets_sample.csv")
        if not sample_path.exists():
            LOGGER.error("Fallback sample %s missing", sample_path)
            return pd.Series(dtype=float)
        df = pd.read_csv(sample_path, parse_dates=["date"])
        mask = df["series"] == series_code
        series = df.loc[mask, ["date", "value"]].set_index("date")["value"].sort_index()
        series.name = series_code.lower()
        return series


def configure_logging(log_level: int = DEFAULT_LOG_LEVEL) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )


def bootstrap_fetcher(config_path: str = "config/config.json") -> DataFetcher:
    """Helper used by notebooks/scripts to instantiate a :class:`DataFetcher`."""
    configure_logging()
    config = FetchConfig.load(Path(config_path))
    return DataFetcher(config=config)


__all__ = [
    "FetchConfig",
    "DataFetcher",
    "bootstrap_fetcher",
    "configure_logging",
]
