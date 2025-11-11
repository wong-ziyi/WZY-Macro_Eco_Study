"""Data acquisition utilities for the Global Liquidity & Productivity Dashboard."""
from __future__ import annotations

import io
import json
import logging
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.error import URLError
from urllib.parse import urljoin

import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:  # pandas-datareader is optional but recommended for macro series
    from pandas_datareader import data as web
    from pandas_datareader import wb
except ImportError:  # pragma: no cover - handled gracefully at runtime
    web = None
    wb = None


LOGGER = logging.getLogger(__name__)
DEFAULT_LOG_LEVEL = logging.INFO
BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = BASE_DATA_DIR / "cache"
SAMPLE_DIR = BASE_DATA_DIR / "sample_data"


def safe_fetch(fetch_func, *args, **kwargs):
    """Call ``fetch_func`` with retries and return a DataFrame on success."""

    description = kwargs.pop("_description", fetch_func.__name__)
    max_attempts = int(kwargs.pop("_max_attempts", 3))
    backoff = float(kwargs.pop("_initial_backoff", 0.5))
    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = fetch_func(*args, **kwargs)
            print(f"[OK] {description} (attempt {attempt})")
            return result
        except Exception as exc:  # pragma: no cover - network dependent
            last_exception = exc
            message = str(exc)
            network_issue = isinstance(
                exc,
                (
                    requests.exceptions.RequestException,
                    URLError,
                    ConnectionError,
                    TimeoutError,
                    socket.gaierror,
                ),
            ) or any(
                token in message.lower()
                for token in ["timed out", "name or service", "temporary failure", "failed to resolve"]
            )
            prefix = "[WARN]" if attempt < max_attempts else "[ERROR]"
            print(f"{prefix} {description} failed on attempt {attempt}/{max_attempts}: {exc}")
            if attempt < max_attempts:
                sleep_for = backoff * (2 ** (attempt - 1))
                time.sleep(sleep_for)
            elif network_issue:
                print(f"[WARN] Network issue detected while fetching {description}")

    if last_exception is not None:
        LOGGER.debug("safe_fetch failed for %s", description, exc_info=last_exception)
    print(f"[INFO] Falling back to empty DataFrame for {description}")
    return pd.DataFrame()


def load_cached_or_sample(name: str) -> pd.DataFrame:
    """Return cached CSV if available, otherwise load sample data."""

    paths = [CACHE_DIR / f"{name}.csv", SAMPLE_DIR / f"{name}.csv"]
    for path in paths:
        if path.exists():
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                print(f"[INFO] Loaded fallback data from {path}")
                return df
            except Exception as exc:  # pragma: no cover - corrupt file
                print(f"[WARN] Failed to load fallback data from {path}: {exc}")
    return pd.DataFrame()


@dataclass
class FetchConfig:
    """Configuration container loaded from ``config/config.json``."""

    raw: Dict
    outputs_dir: Path
    lookback_months: int
    pca_window: int
    regime_states: int
    scenario_shocks: Dict[str, float]
    base_dir: Path

    @classmethod
    def load(cls, config_path: Path) -> "FetchConfig":
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        base_dir = config_path.resolve().parent
        outputs_dir = Path(payload.get("outputs", "outputs"))
        if not outputs_dir.is_absolute():
            outputs_dir = (base_dir / outputs_dir).resolve()
        outputs_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            raw=payload,
            outputs_dir=outputs_dir,
            lookback_months=int(payload.get("lookback_months", 120)),
            pca_window=int(payload.get("pca_window", 36)),
            regime_states=int(payload.get("regime_states", 4)),
            scenario_shocks=payload.get("scenario_shocks", {}),
            base_dir=base_dir,
        )

    def section(self, name: str) -> Dict:
        return self.raw.get(name, {})

    def resolve_path(self, path_str: str) -> Path:
        """Return ``path_str`` as an absolute :class:`Path`."""

        path = Path(path_str)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()


@dataclass
class DataFetcher:
    """Wraps all external data retrieval with safe fallbacks and logging."""

    config: FetchConfig
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=frozenset(["GET"]))
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
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
        if web is None:
            print("[WARN] pandas-datareader not installed; using fallback for Fed balance sheet")
            return self._load_fallback_series("fed_balance_sheet", legacy_code="FED_WALCL")

        def _fetch() -> pd.DataFrame:
            return web.DataReader("WALCL", "fred", start="2000-01-01")

        df = safe_fetch(_fetch, _description="FRED: WALCL")
        if df.empty:
            return self._load_fallback_series("fed_balance_sheet", legacy_code="FED_WALCL")

        series = df.squeeze().astype(float)
        series.index = pd.to_datetime(series.index)
        series.name = "fed_balance_sheet"
        latest = series.dropna().index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] FRED: WALCL fetched (latest={latest_fmt})")
        return series.sort_index()

    def fetch_tga_balance(self) -> pd.Series:
        ticker = self.config.section("creation").get("tga_ticker", "^IRX")

        def _download() -> pd.DataFrame:
            return yf.download(ticker, period="max", interval="1wk", auto_adjust=False, progress=False)

        data = safe_fetch(_download, _description=f"Yahoo Finance: {ticker}")
        if data.empty or "Adj Close" not in data:
            print(f"[WARN] TGA balance fetch failed for {ticker}; using fallback")
            return self._load_fallback_series("tga_balance", legacy_code="TGA_BALANCE")

        series = data["Adj Close"].dropna().rename("tga_balance")
        series.index = pd.to_datetime(series.index)
        latest = series.index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] Yahoo Finance: {ticker} fetched (latest={latest_fmt})")
        return series.sort_index()

    def fetch_ust_issuance(self) -> pd.Series:
        ticker = self.config.section("creation").get("ust_issuance_ticker", "^TNX")

        def _download() -> pd.DataFrame:
            return yf.download(ticker, period="max", auto_adjust=False, progress=False)

        data = safe_fetch(_download, _description=f"Yahoo Finance: {ticker}")
        if data.empty or "Adj Close" not in data:
            print(f"[WARN] UST issuance fetch failed for {ticker}; using fallback")
            return self._load_fallback_series("ust_issuance", legacy_code="UST_ISSUANCE")

        series = data["Adj Close"].dropna().rename("ust_issuance")
        series.index = pd.to_datetime(series.index)
        latest = series.index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] Yahoo Finance: {ticker} fetched (latest={latest_fmt})")
        return series.sort_index()

    def fetch_ecb_total_assets(self) -> pd.Series:
        series_code = self.config.section("creation").get(
            "ecb_series", "BSI/M.U2.N.A.A20.A.1.Z5.0000.Z01.E"
        )
        url = f"https://data.ecb.europa.eu/service/data/{series_code}"

        def _fetch() -> pd.DataFrame:
            response = self.session.get(
                url,
                params={"lastNObservations": "520", "format": "jsondata"},
                headers={"Accept": "application/vnd.sdmx.data+json"},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            series_payload = payload.get("dataSets", [{}])[0].get("series", {})
            if not series_payload:
                raise ValueError("No ECB series payload returned")
            series_key = next(iter(series_payload))
            observations = series_payload.get(series_key, {}).get("observations", {})
            dates = payload.get("structure", {}).get("dimensions", {}).get("observation", [{}])[0].get("values", [])
            records: List[tuple[datetime, float]] = []
            for key, value in observations.items():
                obs_date = dates[int(key)]["id"]
                timestamp = pd.Period(obs_date, freq="M").to_timestamp("M")
                records.append((timestamp, float(value[0])))
            if not records:
                raise ValueError("No ECB observations returned")
            frame = pd.DataFrame(records, columns=["date", "value"]).set_index("date")
            return frame

        df = safe_fetch(_fetch, _description="ECB: total assets")
        if df.empty:
            return self._load_fallback_series("ecb_assets", legacy_code="ECB_ASSETS")

        series = df["value"].astype(float).rename("ecb_assets").sort_index()
        latest = series.dropna().index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] ECB: total assets fetched (latest={latest_fmt})")
        return series

    def fetch_boj_total_assets(self) -> pd.Series:
        base_url = "https://www.boj.or.jp/statistics/boj/other/acmai/release/"
        index_url = urljoin(base_url, "index.htm")

        def _fetch() -> pd.DataFrame:
            index_response = self.session.get(index_url, timeout=30)
            index_response.raise_for_status()
            matches = re.findall(r'href="([^"]*ac(\d+)e\.xlsx)"', index_response.text, flags=re.IGNORECASE)
            if not matches:
                raise ValueError("No BoJ asset workbook links found")
            latest_href, latest_code = max(matches, key=lambda item: item[1])
            workbook_url = urljoin(index_url, latest_href)
            xls_response = self.session.get(workbook_url, timeout=30)
            xls_response.raise_for_status()
            frame = pd.read_excel(io.BytesIO(xls_response.content), index_col=0)
            if frame.empty:
                raise ValueError("BoJ asset workbook empty")
            series = frame.iloc[:, 0].to_frame(name="value")
            series.index = pd.to_datetime(series.index)
            return series

        df = safe_fetch(_fetch, _description="BoJ: total assets")
        if df.empty:
            return self._load_fallback_series("boj_assets", legacy_code="BOJ_ASSETS")

        series = df["value"].astype(float).rename("boj_assets").sort_index()
        latest = series.dropna().index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] BoJ: total assets fetched (latest={latest_fmt})")
        return series

    def fetch_pboc_m2(self) -> pd.Series:
        csv_config = self.config.section("creation").get("pboc_local_csv", "")
        csv_path = self.config.resolve_path(csv_config) if csv_config else Path()
        if csv_path.exists():
            print(f"[OK] Local PBOC M2 CSV loaded from {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            return df.squeeze("columns").rename("pboc_m2")
        print("[WARN] Local PBOC M2 CSV missing; attempting FRED fallback")
        if web is not None:
            def _fetch() -> pd.DataFrame:
                return web.DataReader("MABMM01CNM189S", "fred", start="2000-01-01")

            df = safe_fetch(_fetch, _description="FRED: China M2")
            if not df.empty:
                series = df.squeeze().astype(float).rename("pboc_m2")
                series.index = pd.to_datetime(series.index)
                latest = series.dropna().index.max()
                latest_fmt = latest.date() if pd.notna(latest) else "N/A"
                print(f"[OK] FRED: China M2 fetched (latest={latest_fmt})")
                return series.sort_index()
        print("[INFO] Falling back to sample data for PBOC M2")
        return self._load_fallback_series("pboc_m2", legacy_code="PBOC_M2")

    def fetch_imf_broad_money(self) -> pd.Series:
        countries: Iterable[str] = self.config.section("creation").get("imf_country_codes", [])
        series_frames: List[pd.Series] = []
        for country in countries:
            def _fetch(country_code: str = country) -> pd.DataFrame:
                url = (
                    "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/"
                    f"M.{country_code}.MABMM01_XDC?startPeriod=2000"
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
                    raise ValueError(f"No IMF observations for {country_code}")
                data_map = [
                    (
                        pd.Period(obs["@TIME_PERIOD"], freq="M").to_timestamp("M"),
                        float(obs["@OBS_VALUE"]),
                    )
                    for obs in observations
                ]
                frame = pd.DataFrame(data_map, columns=["date", "value"]).set_index("date")
                return frame

            df = safe_fetch(_fetch, _description=f"IMF: {country}", _max_attempts=1)
            if df.empty:
                print(f"[WARN] IMF request for {country} unavailable; aborting remaining IMF downloads")
                series_frames.clear()
                break
            series_frames.append(df["value"].rename(country))

        if not series_frames:
            print("[WARN] IMF broad money fetch failed for all countries; using fallback")
            return self._load_fallback_series("imf_broad_money", legacy_code="IMF_WORLD")

        combined = pd.concat(series_frames, axis=1).sum(axis=1)
        combined.name = "imf_broad_money"
        latest = combined.dropna().index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] IMF broad money fetched (latest={latest_fmt})")
        return combined.sort_index()

    # ------------------------------------------------------------------
    # FLOW LAYER FETCHERS
    # ------------------------------------------------------------------
    def fetch_market_series(self, tickers: Iterable[str], column_name: str) -> pd.DataFrame:
        tickers = [ticker for ticker in set(tickers) if ticker]
        if not tickers:
            print(f"[WARN] No tickers supplied for {column_name}")
            return pd.DataFrame()

        def _download() -> pd.DataFrame:
            return yf.download(tickers, period="max", auto_adjust=True, progress=False)

        data = safe_fetch(_download, _description=f"Yahoo Finance batch: {','.join(tickers)}")
        if data.empty:
            print(f"[INFO] Falling back to empty DataFrame for {column_name}")
            return pd.DataFrame()

        adj_close = data.get("Adj Close")
        if adj_close is None or adj_close.empty:
            print(f"[WARN] Adjusted close not available for {column_name}")
            return pd.DataFrame()

        if isinstance(adj_close, pd.Series):
            adj_close = adj_close.to_frame(name=tickers[0])

        adj_close = adj_close.loc[:, ~adj_close.columns.duplicated()].dropna(how="all")
        adj_close.columns = pd.Index([f"{column_name}_{col}" for col in adj_close.columns], name="field")
        adj_close.index = pd.to_datetime(adj_close.index)
        latest = adj_close.dropna(how="all").index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] Yahoo Finance batch fetched for {column_name} (latest={latest_fmt})")
        return adj_close

    def fetch_stablecoin_market_cap(self) -> pd.Series:
        endpoint = self.config.section("flow").get("stablecoin_endpoint")
        if not endpoint:
            print("[WARN] Stablecoin endpoint not configured")
            fallback = self._load_fallback_series("stablecoins")
            fallback.name = "stablecoin_market_cap"
            return fallback

        try:
            response = self.session.get(endpoint, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                chart_data = payload.get("totalCharts") or payload.get("charts") or []
            else:
                chart_data = payload or []

            records: Dict[datetime, float] = {}
            iterable = chart_data.items() if isinstance(chart_data, dict) else chart_data
            for point in iterable:
                timestamp: datetime | None = None
                value: float | None = None
                if isinstance(point, tuple) and len(point) == 2:
                    raw_ts, value = point
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    raw_ts, value = point[0], point[1]
                elif isinstance(point, dict):
                    raw_ts = point.get("date") or point.get("timestamp") or point.get("time")
                    value = point.get("total") or point.get("totalCirculatingUSD") or point.get("value")
                else:
                    continue

                if raw_ts is None or value is None:
                    continue

                if isinstance(raw_ts, (int, float)):
                    timestamp = datetime.utcfromtimestamp(raw_ts / (1000 if raw_ts > 10**11 else 1))
                else:
                    try:
                        timestamp = pd.to_datetime(raw_ts).to_pydatetime()
                    except Exception:
                        continue

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue

                records[timestamp] = value

            if not records:
                raise ValueError("No stablecoin market cap data parsed")

            series = pd.Series(dict(sorted(records.items()))).sort_index()
            series.name = "stablecoin_market_cap"
            latest = series.dropna().index.max()
            latest_fmt = latest.date() if pd.notna(latest) else "N/A"
            print(f"[OK] Stablecoin market cap fetched (latest={latest_fmt})")
            return series
        except Exception as exc:
            print(f"[WARN] Stablecoin market cap fetch failed: {exc}")
            fallback = self._load_fallback_series("stablecoins")
            if fallback.empty:
                return pd.Series(dtype=float, name="stablecoin_market_cap")
            fallback.name = "stablecoin_market_cap"
            return fallback

    # ------------------------------------------------------------------
    # UTILIZATION & CONSUMPTION FETCHERS
    # ------------------------------------------------------------------
    def fetch_world_bank_indicator(self, indicator: str) -> pd.Series:
        if wb is None:
            print(f"[WARN] pandas-datareader not installed; cannot fetch World Bank indicator {indicator}")
            return pd.Series(dtype=float, name=indicator)
        try:
            df = wb.download(indicator=indicator, country="all", start=1990, end=datetime.utcnow().year)
            df = df[df[indicator].notna()].groupby("year")[indicator].mean()
            df.index = pd.to_datetime(df.index.astype(str))
            print(f"[OK] World Bank indicator fetched: {indicator}")
            return df.rename(indicator)
        except Exception as exc:
            print(f"[WARN] World Bank fetch failed: {exc}")
            return pd.Series(dtype=float, name=indicator)

    def fetch_oecd_series(self, series: str) -> pd.Series:
        if web is None:
            print(f"[WARN] pandas-datareader not installed; cannot fetch OECD series {series}")
            return pd.Series(dtype=float, name=series)

        mapping = {
            "PMI_NEW_ORDERS": "MEI_PMIO",
            "PMI_INVENTORIES": "MEI_PMII",
            "BCPEBT02": "BCPEBT02",
        }
        resolved = mapping.get(series, series)
        try:
            df = web.DataReader(resolved, "oecd")
            df = df.squeeze("columns").dropna()
            df.index = pd.to_datetime(df.index)
            print(f"[OK] OECD series fetched: {resolved}")
            return df.rename(resolved)
        except Exception as exc:
            print(f"[WARN] OECD fetch failed for {resolved}: {exc}")
            return pd.Series(dtype=float, name=resolved)

    def fetch_retail_sales(self, tickers: Dict[str, str]) -> pd.DataFrame:
        frames = []
        for region, ticker in tickers.items():
            if not ticker:
                continue

            def _download(symbol: str = ticker) -> pd.DataFrame:
                return yf.download(symbol, period="max", auto_adjust=False, progress=False)

            data = safe_fetch(_download, _description=f"Retail sales: {ticker}")
            if data.empty or "Adj Close" not in data:
                print(f"[WARN] Retail sales fetch failed for {ticker}")
                continue

            series = data["Adj Close"].dropna()
            series.index = pd.to_datetime(series.index)
            frames.append(series.rename(region))

        if not frames:
            print("[INFO] No retail sales data fetched")
            return pd.DataFrame()
        combined = pd.concat(frames, axis=1)
        latest = combined.dropna(how="all").index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] Retail sales data fetched (latest={latest_fmt})")
        return combined

    # ------------------------------------------------------------------
    # FALLBACK UTILITIES
    # ------------------------------------------------------------------
    def _load_fallback_series(self, name: str, legacy_code: str | None = None) -> pd.Series:
        frame = load_cached_or_sample(name)
        if not frame.empty:
            if isinstance(frame, pd.Series):
                series = frame.copy()
            elif "value" in frame.columns:
                series = frame["value"]
            else:
                series = frame.iloc[:, 0]
            series.index = pd.to_datetime(series.index)
            series.name = name
            print(f"[INFO] Using fallback data for {name}")
            return series.sort_index()

        sample_map = {
            "PBOC_M2": ("pboc_m2_sample.csv", None),
        }

        if legacy_code:
            file_name, discriminator = sample_map.get(legacy_code, ("cb_assets_sample.csv", "series"))
            sample_path = BASE_DATA_DIR / file_name
            if sample_path.exists():
                df = pd.read_csv(sample_path, parse_dates=["date"])
                if discriminator:
                    mask = df[discriminator] == legacy_code
                    if mask.any():
                        df = df.loc[mask, ["date", "value"]]
                    else:
                        print(f"[WARN] Legacy series {legacy_code} not found in {sample_path}")
                        return pd.Series(dtype=float, name=name)
                else:
                    df = df[["date", "value"]]
                series = df.set_index("date")["value"].sort_index()
                series.name = name
                print(f"[INFO] Using legacy fallback data for {name}")
                return series

        print(f"[WARN] No fallback data available for {name}")
        return pd.Series(dtype=float, name=name)


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
