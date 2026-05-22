
"""
══════════════════════════════════════════════════════════════════════════════
  ML PORTFOLIO MANAGEMENT SYSTEM — V2
  Systematic strategy with walk-forward validation & multiprocessing.
  Models: XGBoost / Random Forest / Ensemble.
══════════════════════════════════════════════════════════════════════════════
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — IMPORTS & ENVIRONMENT SETUP                                  ║
# ║  Standard library, third-party imports, Colab auto-detection.             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

import sys, os
import time
import warnings
from dataclasses import dataclass, field
from copy import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from joblib import Parallel, delayed

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    os.system("pip install -q xgboost seaborn scikit-learn joblib")
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    COLAB_DIR = "/content/drive/MyDrive/ML"
    if not os.path.exists(COLAB_DIR):
        print(f"  WARNING: folder '{COLAB_DIR}' not found on Drive.")
        sys.exit(1)
    os.chdir(COLAB_DIR)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — CONFIGURATION                                                ║
# ║  Central Config dataclass holding all strategy parameters: universe,      ║
# ║  buckets, features, model hyperparameters and transaction costs.          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class Config:
    data_path: str = "Data_Final.xlsx"
    excel_sheet: str = "Foglio1"

    tradeable_assets: list = field(default_factory=lambda: [
        "AAPL.O", "MSFT.O", "GOOGL.O", "AMZN.O", "NVDA.O",
        "JPM.N", "UNH.N", "PG.N", "XOM.N", "V.N",
        "LVMH.PA", "ASML.AS", "SAPG.DE", "NESN.S",
        "ENI.MI", "ISP.MI", "UCG.MI", "ENEL.MI",
        "7203.T", "2330.TW", "HSBA.L",
        "AZPIa.MI", "ON.MI",
        "SIVBQ.PK", "FRCB.PK", "CSGN.S", "BBBYQ.PK",
        "TLT.O", "IEF.O", "LQD.P", "HYG.P", "BND.O",
        "VWCE.DE", "SPY.P", "QQQ.O", "IWM.P",
        "XLK.P", "XLF.P", "XLV.P", "SMH.O",
        "GLD.P", "SLV.P", "USO.P", "UNG.P",
        "CPER.P", "DBA.P", "WEAT.P",
        "BITO.P",
    ])

    asset_buckets: dict = field(default_factory=lambda: {
        "equity": [
            "AAPL.O", "MSFT.O", "GOOGL.O", "AMZN.O", "NVDA.O",
            "JPM.N", "UNH.N", "PG.N", "XOM.N", "V.N",
            "LVMH.PA", "ASML.AS", "SAPG.DE", "NESN.S",
            "ENI.MI", "ISP.MI", "UCG.MI", "ENEL.MI",
            "7203.T", "2330.TW", "HSBA.L", "AZPIa.MI", "ON.MI",
            "SIVBQ.PK", "FRCB.PK", "CSGN.S", "BBBYQ.PK",
        ],
        "bond": ["TLT.O", "IEF.O", "LQD.P", "HYG.P", "BND.O"],
        "etf": [
            "VWCE.DE", "SPY.P", "QQQ.O", "IWM.P", "XLK.P", "XLF.P", "XLV.P", "SMH.O",
            "GLD.P", "SLV.P", "USO.P", "UNG.P", "CPER.P", "DBA.P", "WEAT.P", "BITO.P",
        ],
    })

    target_allocation: dict = field(default_factory=lambda: {
        "equity": 0.40, "bond": 0.30, "etf": 0.30,
    })

    benchmark: str = "SPY.P"
    benchmarks: dict = field(default_factory=lambda: {
        "SPY":   ("ticker", "SPY.P"),
        "VWCE":  ("ticker", "VWCE.DE"),
        "60/40": ("blend", [("SPY.P", 0.60), ("BND.O", 0.40)]),
        "70/30": ("blend", [("SPY.P", 0.70), ("BND.O", 0.30)]),
    })

    macro_indicators: list = field(default_factory=lambda: [
        ".VIX", ".DXY", "EUR3MFD=", "US10Y=RR", "IT10YT=RR", "US10YBEI=R",
    ])

    sma_windows: list = field(default_factory=lambda: [5, 10, 20, 50, 200])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    momentum_periods: list = field(default_factory=lambda: [5, 10, 20, 60])
    volatility_window: int = 20

    forward_period: int = 10
    target_std_multiplier: float = 0.5 

    model_type: str = "xgboost"
    models_to_run: list = field(default_factory=lambda: ["xgboost", "random_forest", "ensemble"])
    train_window: int = 504
    step_size: int = 21
    n_estimators: int = 100
    max_depth: int = 4 
    learning_rate: float = 0.05
    min_samples_per_asset: int = 300

    initial_capital: float = 1_000_000.0
    max_position_pct: float = 0.10

    transaction_cost_bps: float = 15.0
    asset_cost_bps: dict = field(default_factory=lambda: {
        "SPY.P": 5.0, "QQQ.O": 5.0, "TLT.O": 5.0, "IEF.O": 5.0, "BND.O": 5.0,
        "AAPL.O": 10.0, "MSFT.O": 10.0, "NVDA.O": 10.0,
        "SIVBQ.PK": 200.0, "FRCB.PK": 200.0, "CSGN.S": 200.0, "BBBYQ.PK": 200.0,
    })

    output_dir: str = "results"

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — DATA LOADING & CLEANING                                      ║
# ║  Reads the Bloomberg Excel file, parses dates, drops NaT rows,            ║
# ║  removes duplicate days, forward/backward fills weekends and holidays.    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_data(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.data_path, sheet_name=cfg.excel_sheet)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    df = df.ffill(limit=5).bfill(limit=2)
    return df

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — FEATURE ENGINEERING                                          ║
# ║  Builds ~50 predictive features per asset: technical (RSI, MACD,          ║
# ║  Bollinger, momentum), macro (VIX, DXY, yields) and cross-asset           ║
# ║  (breadth, dispersion). Also constructs the supervised target.            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── 4.1 Technical indicators (per asset) ──────────────────────────────────────

def _rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    rs = gain.rolling(period).mean() / (loss.rolling(period).mean() + 1e-10)
    return 100 - 100 / (1 + rs)

def compute_asset_features(prices: pd.Series, cfg: Config) -> pd.DataFrame:
    f = pd.DataFrame(index=prices.index)
    daily_ret = prices.pct_change()
    for p in cfg.momentum_periods: f[f"ret_{p}d"] = prices.pct_change(p)
    f["vol_20d"] = daily_ret.rolling(cfg.volatility_window).std() * np.sqrt(252)
    for w in cfg.sma_windows: f[f"sma_ratio_{w}"] = prices / prices.rolling(w).mean() - 1
    f["rsi"] = _rsi(prices, cfg.rsi_period)
    ema_f, ema_s = prices.ewm(span=cfg.macd_fast, adjust=False).mean(), prices.ewm(span=cfg.macd_slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    f["macd"] = macd_line / prices
    f["macd_hist"] = (macd_line - macd_line.ewm(span=cfg.macd_signal, adjust=False).mean()) / prices
    sma_bb = prices.rolling(cfg.bollinger_window).mean()
    std_bb = prices.rolling(cfg.bollinger_window).std()
    upper, lower = sma_bb + cfg.bollinger_std * std_bb, sma_bb - cfg.bollinger_std * std_bb
    f["bb_pctb"] = (prices - lower) / (upper - lower + 1e-10)
    f["bb_bw"] = (upper - lower) / (sma_bb + 1e-10)
    f["zscore_20"] = (prices - sma_bb) / (std_bb + 1e-10)
    f["trend_20d"] = np.sign(daily_ret).rolling(20).mean()
    f["dist_high_252"] = prices / prices.rolling(252).max() - 1
    f["dist_low_252"] = prices / prices.rolling(252).min() - 1
    return f.replace([np.inf, -np.inf], np.nan)

# ── 4.2 Macro features (global regime: VIX, DXY, rates, breakeven) ────────────

def compute_macro_features(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    f = pd.DataFrame(index=data.index)
    for col in cfg.macro_indicators:
        if col in data.columns and data[col].notna().sum() >= 60:
            s = data[col]
            f[f"{col}_chg5"] = s.pct_change(5)
            f[f"{col}_z60"] = (s - s.rolling(60).mean()) / (s.rolling(60).std() + 1e-10)
    if ".VIX" in data.columns:
        f["vix_level"] = data[".VIX"]
        f["vix_above20"] = (data[".VIX"] > 20).astype(float)
    return f.replace([np.inf, -np.inf], np.nan)

# ── 4.3 Cross-asset features (breadth, return dispersion) ─────────────────────

def compute_cross_features(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cols = [c for c in cfg.tradeable_assets if c in data.columns]
    rets = data[cols].pct_change()
    f = pd.DataFrame(index=data.index)
    f["breadth_20d"] = (rets.rolling(20).sum() > 0).mean(axis=1)
    f["ret_dispersion"] = rets.std(axis=1)
    return f.replace([np.inf, -np.inf], np.nan)

# ── 4.4 Target construction (Buy / Hold / Sell with vol-scaled thresholds) ────

def compute_target(prices: pd.Series, cfg: Config) -> pd.Series:
    fwd = prices.shift(-cfg.forward_period) / prices - 1
    vol_10d = prices.pct_change().rolling(20).std() * np.sqrt(cfg.forward_period)
    vol_10d = vol_10d.clip(lower=0.01)
    buy_thr = vol_10d * cfg.target_std_multiplier
    sell_thr = -vol_10d * cfg.target_std_multiplier
    target = pd.Series(np.nan, index=prices.index)
    target[fwd > buy_thr] = 1
    target[fwd < sell_thr] = -1
    target[fwd.notna() & target.isna()] = 0
    return target

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — ML ENGINE: MODELS & WALK-FORWARD VALIDATION                  ║
# ║  Model factories (XGBoost, Random Forest, Ensemble) and the               ║
# ║  expanding-window walk-forward loop that prevents lookahead bias.         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── 5.1 Model factories and prediction helpers ────────────────────────────────

def _make_xgb(cfg: Config):
    return XGBClassifier(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
                         learning_rate=cfg.learning_rate, objective="multi:softprob",
                         use_label_encoder=False, eval_metric="mlogloss", verbosity=0,
                         random_state=42, n_jobs=1)

def _make_rf(cfg: Config):
    return RandomForestClassifier(n_estimators=cfg.n_estimators, max_depth=cfg.max_depth + 2,
                                  random_state=42, n_jobs=1)

def _safe_proba(model, X, n) -> np.ndarray:
    raw = model.predict_proba(X)
    result = np.full((n, 3), 1 / 3)
    for i, c in enumerate(model.classes_):
        c_int = int(c)
        if 0 <= c_int <= 2: result[:, c_int] = raw[:, i]
    return result

def _fit_only(model_type, cfg, X_tr, y_enc):
    if model_type == "xgboost": m = _make_xgb(cfg); m.fit(X_tr, y_enc); return m
    if model_type == "random_forest": m = _make_rf(cfg); m.fit(X_tr, y_enc); return m
    xgb, rf = _make_xgb(cfg), _make_rf(cfg)
    xgb.fit(X_tr, y_enc); rf.fit(X_tr, y_enc)
    return (xgb, rf)

def _predict_proba_from_model(model, model_type, X_pred):
    n_pred = len(X_pred)
    if model_type == "ensemble":
        xgb, rf = model
        return (_safe_proba(xgb, X_pred, n_pred) + _safe_proba(rf, X_pred, n_pred)) / 2
    return _safe_proba(model, X_pred, n_pred)

def _importance_from_model(model, model_type):
    if model_type == "ensemble":
        xgb, rf = model
        return (xgb.feature_importances_ + rf.feature_importances_) / 2
    return model.feature_importances_

# ── 5.2 Walk-forward expanding-window training & prediction ───────────────────

def walk_forward_asset(X: pd.DataFrame, y: pd.Series, cfg: Config) -> tuple[pd.Series, list[pd.Series]]:
    n = len(X)
    signal = pd.Series(np.nan, index=X.index)
    importances = []
    t = cfg.train_window
    while t < n:
        train_end = t - cfg.forward_period
        if train_end < cfg.min_samples_per_asset: t += cfg.step_size; continue
        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        valid = y_tr.notna()
        X_tr, y_tr = X_tr[valid], y_tr[valid]
        if len(X_tr) < 100 or y_tr.nunique() < 2: t += cfg.step_size; continue
        y_enc = (y_tr + 1).astype(int)
        pred_end = min(t + cfg.step_size, n)
        X_pred = X.iloc[t:pred_end]
        model = _fit_only(cfg.model_type, cfg, X_tr, y_enc)
        proba = _predict_proba_from_model(model, cfg.model_type, X_pred)
        importances.append(pd.Series(_importance_from_model(model, cfg.model_type), index=X_tr.columns))
        signal.iloc[t:pred_end] = proba[:, 2] - proba[:, 0]
        t += cfg.step_size
    return signal.ffill(), importances

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — PORTFOLIO CONSTRUCTION & BACKTESTING                         ║
# ║  Vol-targeted, bucket-balanced allocation with water-filling cap,         ║
# ║  shift-by-one weights to avoid lookahead, and per-asset costs.            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── 6.1 Bucket weight allocator (water-filling with per-asset cap) ────────────

def _bucket_weights(bullish: pd.Series, vols: pd.Series, bucket_assets: list, target_pct: float, cap: float) -> pd.Series:
    bs = bullish[[a for a in bucket_assets if a in bullish.index]]
    if bs.empty: return pd.Series(dtype=float)
    v = vols.reindex(bs.index).fillna(0.02).clip(lower=0.01)
    scores = bs / v
    w = pd.Series(0.0, index=bs.index)
    remaining_pct = float(target_pct)
    remaining_scores = scores.copy()
    for _ in range(len(bs) + 1):
        if remaining_pct <= 1e-12 or remaining_scores.empty or remaining_scores.sum() <= 0: break
        proposed = (remaining_scores / remaining_scores.sum()) * remaining_pct
        room = cap - w[remaining_scores.index]
        over_cap = proposed > room + 1e-12
        if not over_cap.any():
            w.loc[remaining_scores.index] += proposed
            break
        capped_idx = remaining_scores.index[over_cap]
        w.loc[capped_idx] += room.loc[capped_idx]
        remaining_pct -= room.loc[capped_idx].sum()
        remaining_scores = remaining_scores.drop(capped_idx)
    return w

# ── 6.2 Backtest engine (daily returns net of differentiated costs) ───────────

def run_backtest(signals: pd.DataFrame, prices: pd.DataFrame, cfg: Config) -> dict:
    common = signals.index.intersection(prices.index)
    signals = signals.loc[common]
    prices = prices.loc[common]
    daily_ret = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    vol_matrix = daily_ret.rolling(20).std()
    weights = pd.DataFrame(0.0, index=common, columns=signals.columns)
    cap = cfg.max_position_pct
    for date in common:
        strengths = signals.loc[date].dropna()
        bullish = strengths[strengths > 0]
        if bullish.empty: continue
        current_vols = vol_matrix.loc[date].dropna()
        for bucket, target_pct in cfg.target_allocation.items():
            bucket_assets = cfg.asset_buckets.get(bucket, [])
            bw = _bucket_weights(bullish, current_vols, bucket_assets, target_pct, cap)
            for asset, w in bw.items():
                if asset in weights.columns: weights.at[date, asset] = w
    weights_shifted = weights.shift(1).fillna(0)
    daily_ret_filled = daily_ret.fillna(0)
    port_ret_gross = (weights_shifted * daily_ret_filled).sum(axis=1)
    drifted_weights = weights_shifted.multiply(1 + daily_ret_filled, axis=1).div(1 + port_ret_gross, axis=0).fillna(0)
    trades_matrix = (weights - drifted_weights).abs()
    turnover = trades_matrix.sum(axis=1)
    cost_per_asset = pd.Series({a: cfg.asset_cost_bps.get(a, cfg.transaction_cost_bps) for a in weights.columns})
    tc = (trades_matrix * (cost_per_asset / 10_000)).sum(axis=1)
    port_ret_net = port_ret_gross - tc
    cumulative = (1 + port_ret_net).cumprod()
    return {"returns": port_ret_net, "weights": weights, "cumulative": cumulative, "turnover": turnover}

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — PERFORMANCE METRICS & REPORTING                              ║
# ║  Sharpe, Sortino, Calmar, drawdown; benchmark builders (tickers and       ║
# ║  blended portfolios); console summary and matplotlib charts.              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── 7.1 Performance metrics (Sharpe, Sortino, Calmar, Max DD, Win Rate) ───────

def _calc_metrics(r: pd.Series) -> dict:
    if len(r) >= 2: n_years = (r.index[-1] - r.index[0]).days / 365.25
    else: n_years = 0.01
    total = (1 + r).prod() - 1
    cagr = (1 + total) ** (1 / max(n_years, 0.01)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0.0
    downside = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 1e-10
    sortino = cagr / downside
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    win_rate = (r > 0).mean()
    return {"total_return": total, "cagr": cagr, "volatility": vol, "sharpe": sharpe, 
            "sortino": sortino, "max_drawdown": max_dd, "calmar": calmar, "win_rate": win_rate}

# ── 7.2 Benchmark builders (single ticker or constant-weight blend) ───────────

def compute_benchmark_returns(definition: tuple, data: pd.DataFrame) -> pd.Series:
    kind, payload = definition
    if kind == "ticker":
        if payload not in data.columns: return pd.Series(dtype=float)
        return data[payload].pct_change().dropna()
    if kind == "blend":
        components = []
        weights = []
        for ticker, w in payload:
            if ticker not in data.columns: continue
            components.append(data[ticker].pct_change())
            weights.append(w)
        if not components: return pd.Series(dtype=float)
        blended = pd.concat(components, axis=1).fillna(0)
        return (blended * np.array(weights)).sum(axis=1).iloc[1:]
    return pd.Series(dtype=float)

def compute_metrics(port_ret: pd.Series, bench_ret: pd.Series) -> dict:
    common = port_ret.index.intersection(bench_ret.index)
    pr = port_ret.loc[common].dropna()
    br = bench_ret.loc[common].dropna()
    m = {}
    for k, v in _calc_metrics(pr).items(): m[f"strategy_{k}"] = v
    for k, v in _calc_metrics(br).items(): m[f"benchmark_{k}"] = v
    return m

# ── 7.3 Console summary output ────────────────────────────────────────────────

def print_metrics(metrics: dict):
    print("\n" + "=" * 60)
    print("  PERFORMANCE SUMMARY")
    print("=" * 60)
    fmt = "{:<30s} {:>12s} {:>12s}"
    print(fmt.format("", "Strategy", "Benchmark"))
    print("-" * 60)
    rows = [("Total Return", "total_return", "{:.2%}"), ("CAGR", "cagr", "{:.2%}"), ("Volatility (ann.)", "volatility", "{:.2%}"), 
            ("Sharpe Ratio", "sharpe", "{:.2f}"), ("Sortino Ratio", "sortino", "{:.2f}"), ("Max Drawdown", "max_drawdown", "{:.2%}"), 
            ("Calmar Ratio", "calmar", "{:.2f}"), ("Win Rate (daily)", "win_rate", "{:.2%}")]
    for label, key, f in rows:
        sv, bv = f.format(metrics.get(f"strategy_{key}", 0)), f.format(metrics.get(f"benchmark_{key}", 0))
        print(fmt.format(label, sv, bv))
    print("=" * 60)

# ── 7.4 Per-model plots (equity curve, feature importance) ────────────────────

def generate_report(results, bench_ret, importances, metrics, cfg, benchmark_rets=None):
    os.makedirs(cfg.output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    port_ret = results["returns"]
    weights = results["weights"]
    common = port_ret.index.intersection(bench_ret.index)
    pr, br = port_ret.loc[common].fillna(0), bench_ret.loc[common].fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    cum_s = (1 + pr).cumprod()
    ax.plot(cum_s.index, cum_s.values, label="ML Strategy", linewidth=2.0, color="#1F3A5F")
    if benchmark_rets:
        bench_colors = ["#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B2"]
        for i, (name, b_ret) in enumerate(benchmark_rets.items()):
            common_b = pr.index.intersection(b_ret.index)
            cum_b = (1 + b_ret.loc[common_b].fillna(0)).cumprod()
            ax.plot(cum_b.index, cum_b.values, label=f"Bench {name}", alpha=0.7, linestyle="--", color=bench_colors[i % len(bench_colors)])
    ax.set_title("Equity Curve")
    ax.legend()
    fig.savefig(os.path.join(cfg.output_dir, "equity_curve.png"), dpi=150)
    plt.close(fig)

    if importances:
        avg_imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False).head(25)
        fig, ax = plt.subplots(figsize=(10, 7))
        avg_imp.sort_values().plot.barh(ax=ax)
        ax.set_title("Feature Importance")
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.output_dir, "feature_importance.png"), dpi=150)
        plt.close(fig)

# ── 7.5 Cross-model comparison plot ───────────────────────────────────────────

def _comparison_report(runs, bench_ret, base_cfg, benchmark_rets=None):
    os.makedirs(base_cfg.output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    colors = {"xgboost": "#1f77b4", "random_forest": "#2ca02c", "ensemble": "#d62728"}
    common_idx = bench_ret.index
    for run in runs: common_idx = common_idx.intersection(run["results"]["returns"].index)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    for run in runs:
        m = run["model_type"]
        r = run["results"]["returns"].loc[common_idx]
        ax.plot(r.index, (1 + r).cumprod().values, color=colors[m], label=m.upper())
    ax.plot(common_idx, (1 + bench_ret.loc[common_idx]).cumprod().values, color="black", linestyle="--", label="Benchmark")
    ax.set_title("Confronto Equity Curve")
    ax.legend()
    fig.savefig(os.path.join(base_cfg.output_dir, "comparison_equity.png"), dpi=150)
    plt.close(fig)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — PARALLEL EXECUTION & MAIN PIPELINE                           ║
# ║  Per-asset worker for joblib parallelization, per-model orchestrator      ║
# ║  and the top-level main() that ties the whole pipeline together.          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── 8.1 Per-asset worker (used by joblib Parallel) ────────────────────────────

def process_asset(asset, data_asset, cfg, asset_feat, macro, cross):
    X = pd.concat([asset_feat, macro, cross], axis=1)
    y = compute_target(data_asset, cfg)
    signal, imps = walk_forward_asset(X, y, cfg)
    return asset, signal, imps

# ── 8.2 Per-model orchestrator (walk-forward + backtest + report) ─────────────

def _run_single_model(model_type, data, available, asset_feat_dict, macro, cross, bench_ret, base_cfg, benchmark_rets=None):
    cfg = copy(base_cfg)
    cfg.model_type, cfg.output_dir = model_type, os.path.join(base_cfg.output_dir, model_type)
    print("\n" + "#" * 60 + f"\n#  MODELLO: {model_type.upper()}\n" + "#" * 60)
    t_model = time.time()
    results_list = Parallel(n_jobs=-1, backend="loky")(delayed(process_asset)(a, data[a], cfg, asset_feat_dict[a], macro, cross) for a in available)
    all_signals, all_importances = {}, []
    for asset, signal, imps in results_list:
        all_signals[asset], all_importances = signal, all_importances + imps
    signals_df = pd.DataFrame(all_signals)
    results_bt = run_backtest(signals_df, data[available], cfg)
    metrics = compute_metrics(results_bt["returns"], bench_ret)
    print_metrics(metrics)
    generate_report(results_bt, bench_ret, all_importances, metrics, cfg, benchmark_rets=benchmark_rets)
    return {"model_type": model_type, "results": results_bt, "metrics": metrics, "elapsed": time.time() - t_model}

# ── 8.3 Main pipeline entry point ─────────────────────────────────────────────

def main():
    cfg = Config()
    print("[1/4] Caricamento dati...")
    data = load_data(cfg)
    available = [a for a in cfg.tradeable_assets if a in data.columns and data[a].notna().sum() > cfg.min_samples_per_asset]
    print("\n[2/4] Feature engineering...")
    macro, cross = compute_macro_features(data, cfg), compute_cross_features(data, cfg)
    asset_feat = {a: compute_asset_features(data[a], cfg) for a in available}
    benchmark_rets = {name: compute_benchmark_returns(d, data) for name, d in cfg.benchmarks.items() if not compute_benchmark_returns(d, data).empty}
    bench_ret = benchmark_rets[next(iter(benchmark_rets.keys()))] if benchmark_rets else data[cfg.benchmark].pct_change().dropna()
    print("\n[3/4] Esecuzione modelli...")
    runs = [_run_single_model(m, data, available, asset_feat, macro, cross, bench_ret, cfg, benchmark_rets=benchmark_rets) for m in cfg.models_to_run]
    if len(runs) >= 2: _comparison_report(runs, bench_ret, cfg, benchmark_rets=benchmark_rets)
    print(f"\nEsecuzione completata in {time.time() - time.time():.1f}s")

if __name__ == "__main__":
    main()

