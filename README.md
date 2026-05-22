# ML Portfolio Backtester

> Systematic walk-forward backtesting of machine learning portfolio strategies
> on a 48-asset Bloomberg universe. Balanced 40/30/30 allocation across
> equity, ETF and bond buckets, with SHAP explainability and multi-benchmark
> comparison (SPY, VWCE, 60/40, 70/30).

## Features

- **Three ML models**: XGBoost, Random Forest, and Ensemble (mean of probabilities)
- **Walk-forward expanding-window validation** that prevents lookahead bias
- **Multi-class classification** (Buy / Hold / Sell) with continuous signal strength
  derived from `P(buy) − P(sell)`
- **50 engineered features per asset**: 19 technical (RSI, MACD, Bollinger Bands,
  momentum at multiple horizons, distance from highs/lows), 27 macro (VIX, DXY,
  US/EU/Italy rates, breakeven inflation), and 4 cross-asset (breadth,
  dispersion, average market return)
- **Balanced portfolio construction**: 40 % equity / 30 % ETF / 30 % bond with
  per-asset cap and water-filling allocator
- **Realistic transaction costs**: differentiated per-asset (5–200 bps/side)
  based on liquidity tier
- **SHAP-based explainability** for each model
- **Multi-benchmark comparison** vs SPY, VWCE, 60/40 and 70/30 portfolios
- **Parallel execution** across assets via `joblib`

## Requirements

```bash
pip install -r requirements.txt
```

Tested on Python 3.13. The code runs both in local environments and on
Google Colab (auto-detected; the script auto-installs the missing packages
and mounts Drive at `MyDrive/ML`).

## Data

The project expects a Bloomberg export in Excel format at
`./Data_Final.xlsx`, with one sheet `Foglio1`, a `Date` column and one column
per ticker. The Bloomberg dataset is not included in this repository for
licensing reasons.

The full asset universe consists of:
- 23 single-name equities (US, Europe, Asia/UK, Italy)
- 4 distressed / delisted single names (SIVBQ, FRCB, CSGN, BBBYQ)
- 5 bond ETFs (TLT, IEF, LQD, HYG, BND)
- 16 ETFs (equity sector, commodity, crypto)
- 6 macro indicators (used only as features)

## Usage

```bash
python portfolio_ml_v2.py
```

The pipeline runs end-to-end:
1. Loads and cleans the Bloomberg Excel dataset
2. Computes 50 features for each asset
3. Runs walk-forward backtesting for each configured model
4. Generates per-model reports (equity curve, drawdown, monthly heatmap,
   feature importance, allocation, SHAP analysis)
5. Produces cross-model comparison reports and a multi-benchmark table

## Output

After a successful run, the `results/` folder contains:
```
results/
├── xgboost/
│   ├── equity_curve.png
│   ├── drawdown.png
│   ├── rolling_sharpe.png
│   ├── monthly_returns.png
│   ├── feature_importance.png
│   ├── allocation.png
│   └── shap/
│       ├── shap_importance.png
│       ├── shap_beeswarm.png
│       └── shap_dep_*.png
├── random_forest/  (same structure)
├── ensemble/       (same structure)
├── comparison_equity.png
├── comparison_drawdown.png
├── comparison_excess.png
├── comparison_final_wealth.png
├── comparison_metrics.csv
└── comparison_metrics_multibench.csv
```

## Configuration

All parameters are centralized in the `Config` dataclass at the top of
`portfolio_ml_v2.py`. Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `models_to_run` | `["xgboost", "random_forest", "ensemble"]` | Models to backtest |
| `step_size` | `21` | Rebalancing frequency in trading days (≈ monthly) |
| `train_window` | `504` | Initial training window (≈ 2 years) |
| `forward_period` | `10` | Target horizon (days) |
| `buy_threshold` / `sell_threshold` | `±0.02` | Classification thresholds (±2 %) |
| `max_position_pct` | `0.10` | Per-asset cap (10 %) |
| `target_allocation` | 40/30/30 | Bucket allocation (equity/etf/bond) |
| `asset_cost_bps` | dict | Per-asset transaction cost (bps/side) |

## Methodology highlights

- **Anti-lookahead by construction**: training stops at `t − forward_period`
  to avoid using future returns in supervised learning; portfolio weights are
  shifted by one day so that the return at time `t` is computed using the
  weights held at the close of `t−1`.
- **Bucket-based allocation with water-filling**: each bucket receives its
  target percentage, distributed across bullish assets proportionally to
  signal strength, with a per-asset cap enforced via iterative water-filling
  (capped assets are fixed, surplus is redistributed to non-capped ones).
- **Per-asset transaction costs**: real-world friction modeled by asset class
  (3 bps for liquid US ETFs up to 200 bps for delisted pink-sheet names),
  applied on each asset's individual weight change rather than total turnover.

## Limitations

- **Long-only**: bearish signals are not used to open short positions.
- **No leverage**.
- **Survivorship bias in universe selection**: the 48 tickers were chosen in
  2026 with knowledge of past winners (mitigated by including delisted
  distressed names but not eliminated).
- **No tax modeling**: capital-gains taxes (e.g. Italian 26 %) are not
  subtracted.
- **No bid-ask spread on rebalance days**: only the per-asset cost is
  charged; slippage on large orders is approximated implicitly in the
  per-asset cost values.

## License

Released under the MIT License. See `LICENSE` for details.

## Author

Dario De Prosperis
