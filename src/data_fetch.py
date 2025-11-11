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

ECB_API = "https://data-api.ecb.europa.eu/service/data"
OECD_BASE = "https://stats.oecd.org/sdmx-json/data"
FRED_RETAIL = {
    "US": "RSAFS",
    "EU": "RSIEURO",
    "CN": "CHNRSCYOY",
    "IN": "INDRSTYOY",
}


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


def pick_close(df: pd.DataFrame) -> pd.Series:
    """Select the best available close column from a Yahoo Finance frame."""

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join([c for c in col if c]) for col in df.columns.values]
    for col in ["Adj Close", "Adjusted Close", "Close", "close"]:
        if col in df.columns:
            series = df[col].copy()
            series.name = "close"
            return series
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        return df[num_cols[0]].rename("close")
    return pd.Series(dtype=float, name="close")


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


def fetch_ecb_total_assets(last_n: int = 520) -> pd.DataFrame:
    """Retrieve Eurosystem or broader ECB balance sheet totals."""

    ilm_key = "ILM.W.U2.C.T000000.Z5.Z01"
    bsi_key = "BSI.M.U2.N.A.A20.A.1.U2.0000.Z01.E"

    for key in (ilm_key, bsi_key):
        url = f"{ECB_API}/{key}?lastNObservations={last_n}&format=jsondata"
        try:
            df = safe_fetch(pd.read_json, url, _description=f"ECB JSON {key}", _max_attempts=1)
            if isinstance(df, pd.DataFrame) and not df.empty:
                if "date" in df.columns and "value" in df.columns:
                    frame = df.copy()
                    frame["date"] = pd.to_datetime(frame["date"])
                    frame = frame.set_index("date").sort_index()
                else:
                    frame = df
                if not frame.empty:
                    latest = frame.dropna().index.max() if isinstance(frame.index, pd.DatetimeIndex) else None
                    latest_fmt = latest.date() if isinstance(latest, pd.Timestamp) else "N/A"
                    print(f"[OK] ECB: {key} fetched (latest={latest_fmt}, rows={len(frame)})")
                    return frame

            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            obs = []
            series = data.get("data", {}).get("dataSets", [{}])[0].get("series", {})
            dims = data.get("data", {}).get("structure", {}).get("dimensions", {}).get("observation", [])
            if not dims:
                raise ValueError("No observation dimensions present")
            time_values = [v.get("name") for v in dims[0].get("values", [])]
            for _, s_val in series.items():
                for obs_key, obs_val in s_val.get("observations", {}).items():
                    idx = int(obs_key)
                    if idx >= len(time_values):
                        continue
                    timestamp = pd.to_datetime(time_values[idx])
                    value = obs_val[0] if isinstance(obs_val, (list, tuple)) else obs_val
                    obs.append((timestamp, float(value)))
            frame = pd.DataFrame(obs, columns=["date", "value"]).set_index("date").sort_index()
            if not frame.empty:
                latest = frame.index.max()
                latest_fmt = latest.date() if pd.notna(latest) else "N/A"
                print(f"[OK] ECB: {key} fetched (latest={latest_fmt}, rows={len(frame)})")
                return frame
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[WARN] ECB: {key} failed: {exc}")

    print("[INFO] Falling back to cached/sample ecb_assets")
    return load_cached_or_sample("ecb_assets")


def fetch_china_broad_money() -> pd.DataFrame:
    """Fetch China broad money (M3) series from FRED."""

    if web is None:
        print("[WARN] pandas-datareader not installed; cannot fetch China broad money")
        return load_cached_or_sample("china_broad_money")

    try_ids = ["MABMM301CNM189S", "MABMM301CNA189S"]
    for sid in try_ids:
        try:
            df = safe_fetch(
                web.DataReader,
                sid,
                "fred",
                start="1999-01-01",
                _description=f"FRED China broad money {sid}",
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                frame = df.rename(columns={sid: "value"}).sort_index()
                latest = frame.dropna().index.max()
                latest_fmt = latest.date() if pd.notna(latest) else "N/A"
                print(
                    f"[OK] FRED China broad money: {sid} (latest={latest_fmt}, rows={len(frame)})"
                )
                return frame
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[WARN] FRED China broad money failed for {sid}: {exc}")

    print("[INFO] Falling back to cached/sample china_broad_money")
    return load_cached_or_sample("china_broad_money")


def fetch_oecd(dataset: str, filter_path: str = "all/all", params: str = "contentType=csv") -> pd.DataFrame:
    """Retrieve OECD SDMX datasets via HTTPS."""

    url = f"{OECD_BASE}/{dataset}/{filter_path}?{params}"
    try:
        df = safe_fetch(pd.read_csv, url, _description=f"OECD: {dataset}", _max_attempts=1)
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"[OK] OECD: {dataset} fetched (rows={len(df)})")
            return df
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[WARN] OECD fetch failed for {dataset}: {exc}")

    print(f"[INFO] Falling back to cached/sample oecd_{dataset.lower()}")
    return load_cached_or_sample(f"oecd_{dataset.lower()}")


def fetch_retail_sales(series_map: Dict[str, str] | None = None) -> Dict[str, pd.Series]:
    """Fetch retail sales series from FRED with graceful fallbacks."""

    if series_map is None:
        series_map = FRED_RETAIL

    series_map = {k: v for k, v in (series_map or {}).items() if v}
    if not series_map:
        print("[INFO] No retail sales symbols provided")
        return {}

    results: Dict[str, pd.Series] = {}
    if web is None:
        print("[WARN] pandas-datareader not installed; using fallback retail sales data")
        for region in series_map:
            fb = load_cached_or_sample(f"retail_{region.lower()}")
            if not fb.empty:
                if isinstance(fb, pd.Series):
                    series = fb.copy()
                elif "value" in fb.columns:
                    series = fb["value"]
                else:
                    series = fb.iloc[:, 0]
                series.name = region
                results[region] = series.sort_index()
        return results

    from pandas_datareader._utils import RemoteDataError  # type: ignore

    for region, sid in series_map.items():
        try:
            df = safe_fetch(
                web.DataReader,
                sid,
                "fred",
                start="2000-01-01",
                _description=f"Retail sales {region}/{sid}",
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                frame = df.rename(columns={sid: "value"}).sort_index()
                series = frame["value"].astype(float)
                latest = series.dropna().index.max()
                latest_fmt = latest.date() if pd.notna(latest) else "N/A"
                print(
                    f"[OK] Retail (FRED): {region}/{sid} (latest={latest_fmt}, rows={len(series)})"
                )
                results[region] = series
                continue
        except (Exception, RemoteDataError) as exc:  # pragma: no cover - network dependent
            print(f"[WARN] Retail sales fetch failed for {region}/{sid}: {exc}")

        fb = load_cached_or_sample(f"retail_{region.lower()}")
        if not fb.empty:
            if isinstance(fb, pd.Series):
                series = fb.copy()
            elif "value" in fb.columns:
                series = fb["value"]
            else:
                series = fb.iloc[:, 0]
            series = series.sort_index()
            series.name = region
            print(f"[INFO] Using fallback retail for {region}")
            results[region] = series

    if not results:
        print("[INFO] No retail sales data fetched")
    return results


_fetch_retail_sales_helper = fetch_retail_sales


def fetch_yf_batch(symbols: Iterable[str], start: str = "2000-01-01") -> Dict[str, pd.Series]:
    """Batch download Yahoo Finance series and return close columns."""

    symbols = [s.strip() for s in symbols if isinstance(s, str) and s.strip()]
    if not symbols:
        print("[INFO] Yahoo Finance: empty symbol list, skipping")
        return {}

    try:
        data = safe_fetch(
            yf.download,
            tickers=" ".join(symbols),
            start=start,
            progress=False,
            auto_adjust=False,
            _description=f"Yahoo Finance batch {','.join(symbols)}",
        )
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[WARN] Yahoo Finance batch failed: {exc}")
        return {}

    results: Dict[str, pd.Series] = {}
    if isinstance(data, pd.DataFrame) and not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            available = set(data.columns.get_level_values(-1))
            for symbol in symbols:
                if symbol not in available:
                    continue
                sub = data.xs(symbol, axis=1, level=-1, drop_level=True)
                series = pick_close(sub)
                if not series.empty:
                    results[symbol] = series.sort_index()
        else:
            series = pick_close(data)
            if not series.empty:
                results[symbols[0]] = series.sort_index()

    if results:
        latest_candidates = [s.dropna().index.max() for s in results.values() if not s.dropna().empty]
        latest = max(latest_candidates) if latest_candidates else None
        latest_fmt = latest.date() if isinstance(latest, pd.Timestamp) else "N/A"
        rows = sum(len(s.dropna()) for s in results.values())
        print(
            f"[OK] Yahoo Finance batch: {','.join(results.keys())} (latest={latest_fmt}, rows={rows})"
        )
    else:
        print(f"[WARN] Yahoo Finance batch returned no data for {','.join(symbols)}")

    return results


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
        batch = fetch_yf_batch([ticker])
        series = batch.get(ticker)
        if series is None or series.empty:
            print(f"[WARN] TGA balance fetch failed for {ticker}; using fallback")
            return self._load_fallback_series("tga_balance", legacy_code="TGA_BALANCE")

        result = series.rename("tga_balance").sort_index()
        latest = result.dropna().index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] TGA balance proxy fetched (latest={latest_fmt}, rows={len(result)})")
        return result

    def fetch_ust_issuance(self) -> pd.Series:
        ticker = self.config.section("creation").get("ust_issuance_ticker", "^TNX")
        batch = fetch_yf_batch([ticker])
        series = batch.get(ticker)
        if series is None or series.empty:
            print(f"[WARN] UST issuance fetch failed for {ticker}; using fallback")
            return self._load_fallback_series("ust_issuance", legacy_code="UST_ISSUANCE")

        result = series.rename("ust_issuance").sort_index()
        latest = result.dropna().index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] UST issuance proxy fetched (latest={latest_fmt}, rows={len(result)})")
        return result

    def fetch_ecb_total_assets(self) -> pd.Series:
        df = fetch_ecb_total_assets()
        if df.empty:
            return self._load_fallback_series("ecb_assets", legacy_code="ECB_ASSETS")

        if "value" in df.columns:
            series = df["value"].astype(float)
        elif isinstance(df, pd.Series):
            series = df.astype(float)
        else:
            first_numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if first_numeric:
                series = df[first_numeric[0]].astype(float)
            else:
                return self._load_fallback_series("ecb_assets", legacy_code="ECB_ASSETS")

        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)
        series = series.rename("ecb_assets").sort_index()
        latest = series.dropna().index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] ECB total assets prepared (latest={latest_fmt}, rows={len(series)})")
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
        df = fetch_china_broad_money()
        if df.empty:
            print("[INFO] Falling back to sample data for PBOC M2")
            return self._load_fallback_series("pboc_m2", legacy_code="PBOC_M2")

        series = df["value"] if "value" in df.columns else df.iloc[:, 0]
        if not isinstance(series, pd.Series):
            series = pd.Series(dtype=float, name="pboc_m2")
        series = series.astype(float)
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)
        series = series.rename("pboc_m2").sort_index()
        latest = series.dropna().index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] China broad money prepared (latest={latest_fmt}, rows={len(series)})")
        return series

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
        tickers = [ticker.strip() for ticker in set(tickers) if isinstance(ticker, str) and ticker.strip()]
        if not tickers:
            print(f"[WARN] No tickers supplied for {column_name}")
            return pd.DataFrame()

        data_map = fetch_yf_batch(tickers)
        if not data_map:
            print(f"[INFO] Falling back to empty DataFrame for {column_name}")
            return pd.DataFrame()

        frames = []
        for symbol, series in data_map.items():
            if series.empty:
                continue
            frames.append(series.rename(f"{column_name}_{symbol}"))

        if not frames:
            print(f"[INFO] No usable Yahoo Finance data for {column_name}")
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1).sort_index()
        combined.index = pd.to_datetime(combined.index)
        latest = combined.dropna(how="all").index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(
            f"[OK] Market series prepared for {column_name} (latest={latest_fmt}, rows={len(combined)})"
        )
        return combined

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
        dataset_map = {
            "PMI_NEW_ORDERS": "MEI_PMIO",
            "PMI_INVENTORIES": "MEI_PMII",
            "BCPEBT02": "BCPEBT02",
        }
        dataset = dataset_map.get(series, series)
        df = fetch_oecd(dataset)
        if df.empty:
            print(f"[WARN] OECD dataset empty for {dataset}")
            return pd.Series(dtype=float, name=dataset)

        frame = df.copy()
        for column in ["LOCATION", "SUBJECT", "MEASURE", "FREQUENCY"]:
            if column in frame.columns and frame[column].nunique() > 1:
                value = frame[column].dropna().iloc[0]
                frame = frame[frame[column] == value]

        time_col = next(
            (col for col in ["TIME_PERIOD", "TIME", "Date", "date"] if col in frame.columns),
            None,
        )
        value_col = next(
            (col for col in ["ObsValue", "OBS_VALUE", "Value", "value"] if col in frame.columns),
            None,
        )

        if time_col is None or value_col is None:
            numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
            if not numeric_cols:
                print(f"[WARN] Could not determine OECD value column for {dataset}")
                return pd.Series(dtype=float, name=dataset)
            value_col = numeric_cols[0]
            time_col = time_col or frame.columns[0]

        frame[time_col] = pd.to_datetime(frame[time_col])
        frame = frame.dropna(subset=[value_col]).sort_values(time_col)
        series_out = frame.set_index(time_col)[value_col].astype(float)
        series_out.name = dataset
        latest = series_out.dropna().index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] OECD series prepared: {dataset} (latest={latest_fmt}, rows={len(series_out)})")
        return series_out

    def fetch_retail_sales(self, tickers: Dict[str, str]) -> pd.DataFrame:
        series_map = {region: ticker for region, ticker in (tickers or {}).items() if ticker}
        data_map = _fetch_retail_sales_helper(series_map if series_map else None)
        if not data_map:
            print("[INFO] No retail sales data fetched")
            return pd.DataFrame()

        frames = []
        for region, series in data_map.items():
            if series.empty:
                continue
            frames.append(series.rename(region))

        if not frames:
            print("[INFO] Retail sales inputs empty after fetch")
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1).sort_index()
        latest = combined.dropna(how="all").index.max()
        latest_fmt = latest.date() if pd.notna(latest) else "N/A"
        print(f"[OK] Retail sales data prepared (latest={latest_fmt}, rows={len(combined)})")
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
    "pick_close",
    "fetch_ecb_total_assets",
    "fetch_china_broad_money",
    "fetch_oecd",
    "fetch_retail_sales",
    "fetch_yf_batch",
]
