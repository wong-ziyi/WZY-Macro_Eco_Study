# Data Dictionary — Global Liquidity & Productivity Dashboard

## Inputs

| Layer | Variable | Source | Frequency | Notes/Fallback |
|-------|----------|--------|-----------|----------------|
| Creation | `fed_balance_sheet` | FRED via yfinance (`WALCL`) | Weekly (resampled monthly) | Falls back to `data/cb_assets_sample.csv` |
| Creation | `ecb_assets` | ECB SDW REST (`BSI/...`) | Monthly | Fallback sample if request fails |
| Creation | `boj_assets` | BoJ XLS (URL configurable) | Monthly | Fallback sample |
| Creation | `pboc_m2` | Local CSV (`data/pboc_m2_sample.csv`) | Annual | Extend via forward-fill |
| Creation | `imf_broad_money` | IMF IFS SDMX JSON | Monthly | Aggregated sum across configured economies |
| Creation | `tga_balance` | FRED via yfinance (`WTREGEN`) | Weekly | Proxy for US Treasury General Account |
| Creation | `ust_issuance` | yfinance ticker (`USTBOND`) | Daily | Change in US Treasury issuance |
| Flow | `fx` | yfinance (`DX-Y.NYB`) | Daily | DXY inverted for flow index |
| Flow | `rates` | yfinance (`^TNX`) | Daily | 10Y Treasury yield inverted |
| Flow | `credit` | yfinance (`BAMLH0A0HYM2`) | Daily | High-yield OAS inverted |
| Flow | `equities` | yfinance (SPY, QQQ, etc.) | Daily | Equal-weighted log returns |
| Flow | `bonds` | yfinance (AGG, EMB) | Daily | Equal-weighted log returns |
| Flow | `commodities` | yfinance (CL=F, GC=F) | Daily | Optional weighting |
| Flow | `stablecoin_market_cap` | DefiLlama | Daily | Total stablecoin circulation |
| Utilization | `world_bank_gfcf` | World Bank (`NE.GDI.FTOT.ZS`) | Annual | Mean across countries |
| Utilization | `oecd_capex` | OECD API (`BCPEBT02`) | Quarterly | Resampled to monthly |
| Utilization | `pmi_spread` | Configured PMI series | Monthly | New orders minus inventories |
| Consumption | `retail_sales` | yfinance proxies | Monthly | YoY growth |
| Consumption | `global_services_pmi` | Configured PMI | Monthly | |
| Consumption | `real_wages` | OECD/other | Quarterly | |
| Efficiency | `pwt_tfp` | Penn World Table | Annual | | 
| Efficiency | `rd_world_bank` | World Bank (`GB.XPD.RSDV.GD.ZS`) | Annual | | 
| Efficiency | `semi_sales` | Local CSV sample | Monthly | |

## Derived outputs

| Output | Description | Location |
|--------|-------------|----------|
| `creation_index.csv` | Creation layer components and Z-scores | `outputs/creation_index.csv` |
| `flow_index.csv` | Weighted flow index and constituent Z-scores | `outputs/flow_index.csv` |
| `absorption_index.csv` | Utilization, consumption, absorption Z-scores | `outputs/absorption_index.csv` |
| `productivity_momentum.csv` | Efficiency inputs and composite | `outputs/productivity_momentum.csv` |
| `composite_index.csv` | Weighted composite plus components | `outputs/composite_index.csv` |
| `rolling_pca.csv` | Leading component scores | `outputs/rolling_pca.csv` |
| `regimes.csv` | Markov-switching regime probabilities | `outputs/regimes.csv` |
| `scenario_tree.csv` | Scenario stress table | `outputs/scenario_tree.csv` |
| `creation_index.png` | Creation index chart | `outputs/creation_index.png` |
| `flow_index.png` | Flow index chart | `outputs/flow_index.png` |
| `absorption_index.png` | Absorption chart | `outputs/absorption_index.png` |
| `productivity_momentum.png` | Productivity chart | `outputs/productivity_momentum.png` |
| `composite_index.png` | Composite dashboard line chart | `outputs/composite_index.png` |
| `four_quadrant_panel.png` | Creation vs Absorption scatter with quadrants | `outputs/four_quadrant_panel.png` |
| `regional_heatmap.png` | ETF region heatmap | `outputs/regional_heatmap.png` |
| `regional_heatmap.csv` | Heatmap table | `outputs/regional_heatmap.csv` |
| `traffic_light_dashboard.png` | Traffic-light dashboard | `outputs/traffic_light_dashboard.png` |
| `traffic_light_status.json` | JSON summary for dashboard | `outputs/traffic_light_status.json` |

All generated CSVs use ISO 8601 timestamps; charts are saved at 300 DPI.
