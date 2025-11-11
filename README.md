# Global Liquidity & Productivity Dashboard — Free Data Edition

The Global Liquidity & Productivity Dashboard (GLPD) monitors the world-wide liquidity cycle using entirely free data sources. The project is engineered for reproducible execution on Google Colab or any Python 3.10+ environment.

## Quick start (Colab or local)

1. Clone the repository and install dependencies:
   ```bash
   git clone <repo-url>
   cd WZY-Macro_Eco_Study
   pip install -r requirements.txt
   ```

2. Launch the automation notebook:
   ```bash
   jupyter notebook notebooks/run_dashboard.ipynb
   ```
   or open the notebook directly in Google Colab.

3. Run all cells to execute the end-to-end pipeline:
   - Fetch latest macro & market data (with built-in fallbacks for offline use).
   - Compute liquidity, flow, absorption, and productivity indices.
   - Generate composite metrics, rolling PCA, regime detection, and scenario analysis.
   - Export CSV tables and matplotlib charts to `outputs/`.

## Repository layout

```
├── config/config.json     # Centralized tickers, endpoints, weights, and output settings
├── data/                  # Local fallback samples (PBOC M2, semiconductor sales, etc.)
├── notebooks/run_dashboard.ipynb
├── outputs/               # Generated figures & tables
├── src/
│   ├── data_fetch.py      # Free-data ingestion with graceful degradation
│   ├── compute_indices.py # Index calculations, PCA, regimes, scenarios
│   └── visualize.py       # Matplotlib charting utilities
├── data_dictionary.md     # Field & output reference
├── requirements.txt
└── README.md
```

## Reproducibility notes

- All remote calls use free APIs (FRED via yfinance, ECB SDW, IMF SDMX JSON, DefiLlama, World Bank, OECD, Penn World Table).
- Each fetcher provides detailed logging and local CSV fallbacks to support offline execution.
- Charts are created using plain matplotlib and saved as PNG files in `outputs/`.
- The notebook orchestrates the workflow and is safe to re-run; intermediate data is cached only in memory.

## Extensions implemented

- Rolling PCA to estimate a common global liquidity factor.
- Markov-regime detection (4-state) to classify macro regimes (expansion, reflation, contraction, liquidity trap).
- Scenario tree to stress-test composite index under rate, FX, and stablecoin shocks.

## License

This project aggregates public, open datasets. Ensure compliance with each provider’s terms of use when redistributing derived outputs.
