# AlphaScout — Quantitative Momentum Strategy

**Kalman Filter · OU Mean Reversion · Half-Kelly · GPU-Accelerated ML**

---

## Architecture

```
alphascout_backtest.py   Pure-math strategy + backtester
alphascout_ml.py         GPU ML layer (XGBoost + LSTM + Adaptive Kalman)
alphascout_live.py       Alpaca paper/live trading with ML signals
alphascout_tradingview.pine  Pine Script v5 single-instrument strategy
```

All three files are standalone and share no state at rest. The live script
imports from the other two. The backtest works fully without the ML module.

---

## Mathematical Components

### 1 · Kalman Filter (2-state)

State `x = [level, velocity]ᵀ`. Applied to **log-prices**.

```
Predict   x⁻ = Fx,        P⁻ = FPFᵀ + Q
Update    K  = P⁻Hᵀ / S,  x  = x⁻ + K·ν,  P = (I − KH)P⁻
```

`F = [[1,1],[0,1]]`, `H = [[1,0]]`, `Q = diag([q_L, q_V])`.
The **velocity** component is the primary trend signal.
The noise ratio `R/q_V` controls smoothness vs responsiveness.

### 2 · Ornstein-Uhlenbeck Process

```
dXₜ = θ(μ − Xₜ)dt + σ dWₜ
```

Parameters estimated via OLS on the Kalman residuals (log-price minus
KF level). Gives a **statistically grounded z-score** for mean-reversion
entry: `z = (Xₜ − μ̂) / σ_eq`, `σ_eq = σ̂ / √(2θ̂)`.
Half-life `τ = ln 2 / θ̂` measures how fast the deviation decays.

### 3 · Half-Kelly Criterion

Continuous Kelly for normally distributed returns: `f* = μ/σ²`.
Half-Kelly `f = f*/2` cuts variance by 75% while retaining ~94% of the
expected log-growth rate. Rolling 60-day μ̂, σ̂² adapt to changing dynamics.
Position fractions capped at `MAX_POS_FRAC = 0.25`.

### 4 · Sharpe Momentum

`SR_i = (μ̂_i · 252) / (σ̂_i · √252)` over 20-day rolling window.
Penalises high-vol stocks — selects cleaner, more persistent trends
rather than raw return momentum.

### 5 · EWMA-GARCH Volatility Regime

`σ²ₜ = λ σ²ₜ₋₁ + (1−λ) rₜ²`, λ = 0.94 (RiskMetrics).
Regime scalar ∈ {0.50, 0.75, 1.00, 1.25} scales all position sizes.

---

## ML Components (GPU)

### Walk-Forward XGBoost

Binary classifier: `y_t = 𝟙[close_{t+5} > close_t]`.
Trained on an **expanding window**, retrained every 21 bars.
13 features: Kalman velocity, OU z-score, Kelly fraction, Sharpe momentum,
vol regime, MACD, volume ratio, returns, volatility, ATR.
GPU: `device='cuda'`, histogram split algorithm.

**No lookahead**: features at `t` use only data ≤ `t`; labels require
`close_{t+5}` which is excluded from training until it is observed.

### LSTM Regime Classifier

```
60-day SPY sequence → 2-layer LSTM (128 hidden) → P(bull) ∈ [0,1]
```

Label: `𝟙[SPY return_{t+20} > 0]` — purely data-driven, no subjective
regime definition. Replaces the simple Kalman-velocity threshold.
Trained with AdamW + cosine LR + early stopping. PyTorch CUDA.

### Adaptive Kalman Filter (Deep SSM)

Learns `Q_t = f_Q(context; θ)` and `R_t = f_R(context; θ)` via a
small MLP, making noise parameters respond to market conditions.

Training loss — negative log marginal likelihood (exact MLE):
```
L(θ) = ½ Σₜ [ νₜ²/Sₜ + log Sₜ ]
```

Kalman recursion implemented as differentiable torch ops so gradients
flow through all timesteps. Enabled via `use_akf=True` in `MLPipeline`.

---

## Scoring System

| Component | Points | Scale |
|---|---|---|
| Kalman velocity / price above KF / vol regime | 0–3 | base |
| OU z-score entry zone | 0–2 | base |
| Sharpe momentum (>0.5 and >1.0) | 0–2 | base |
| MACD cross | 0–1 | base |
| Volume confirmation | 0–1 | base |
| XGBoost P(return>0) >0.55/0.60/0.70 | 0–3 | ML bonus |
| **Total** | **0–12** | |

Entry threshold: **≥ 5** (base only) or **≥ 7** (with ML).

---

## Data Delay Model

```
Signal computed at bar t  →  order fills at bar t + DATA_DELAY_BARS
Simulated as: fill_price = close[t + delay] × (1 ± cost_bps/10000)
```

| Data source | Recommended setting |
|---|---|
| Free EOD (yfinance / Alpaca IEX) | `DATA_DELAY_BARS = 1` |
| Extra conservative | `DATA_DELAY_BARS = 2` |

Total round-trip cost: 16 bps (10 commission + 4 spread + 2 impact).

---

## On "1% Daily Returns"

`1%/day = (1.01²⁵² − 1) ≈ 1,147% annual.`

No strategy sustains this. Renaissance Medallion (best ever) averages
~66% net annually. This strategy targets **25–50% CAGR** with
**Sharpe > 1.5** — placing it in the top 1% of all investment vehicles.

---

## Setup

```bash
pip install yfinance alpaca-py torch xgboost scikit-learn \
            scipy schedule matplotlib pandas numpy
```

### API Keys

```bash
export ALPACA_API_KEY="your-key-id"
export ALPACA_API_SECRET="your-secret-key"
```

Or edit the `API_KEY` / `API_SECRET` variables at the top of
`alphascout_live.py`.

Get free Alpaca paper-trading keys at: https://alpaca.markets

---

## Workflow

### Step 1 — Pure-math backtest (no ML)

```bash
python alphascout_backtest.py
```

Produces `alphascout_backtest.png` with equity curve, drawdown,
monthly returns heatmap, and performance table.

### Step 2 — ML-enhanced backtest

```bash
python alphascout_backtest.py --ml
```

- Trains XGBoost (per-ticker, walk-forward) + LSTM on SPY history
- Saves models to `./alphascout_models/`
- Produces `alphascout_ml_backtest.png`
- First run takes ~10–30 min depending on GPU / universe size

### Step 3 — Check account status

```bash
python alphascout_live.py --status
```

### Step 4 — Single live cycle (paper)

```bash
python alphascout_live.py --now
```

Runs one complete cycle: fetch → features → ML → score → trade.

### Step 5 — Train ML from scratch then run

```bash
python alphascout_live.py --train --now
```

### Step 6 — Start daily scheduler

```bash
python alphascout_live.py --schedule
```

Runs every weekday at 16:15 ET (15 min after close, data settled).

### Step 7 — Go live (when satisfied with paper results)

```bash
python alphascout_live.py --live --schedule
```

⚠ This trades real money. Ensure paper results are satisfactory first.

---

## GPU Notes

The strategy auto-detects CUDA:

```python
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
XGB_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

GPU is used for:
- XGBoost histogram split-finding (`device='cuda'`)
- LSTM forward/backward passes (`model.to(DEVICE)`)
- Adaptive Kalman differentiable recursion (all torch ops)

Typical speedup vs CPU: **5–15× for XGBoost**, **10–30× for LSTM**.
The strategy runs correctly on CPU if CUDA is unavailable.

To confirm GPU is detected:

```python
import torch, xgboost as xgb
print(torch.cuda.is_available())            # True
print(torch.cuda.get_device_name(0))        # e.g. NVIDIA RTX 4090
```

---

## Configuration Reference

### `alphascout_backtest.py`

| Parameter | Default | Description |
|---|---|---|
| `BACKTEST_START` | `2020-01-01` | Backtest start date |
| `BACKTEST_END` | `2025-05-30` | Backtest end date |
| `INITIAL_CAPITAL` | `100,000` | Starting equity |
| `MAX_POSITIONS` | `5` | Max concurrent holdings |
| `MAX_POS_FRAC` | `0.25` | Max fraction per stock |
| `DATA_DELAY_BARS` | `1` | Signal-to-fill bar delay |
| `COST_BPS` | `16` | Round-trip cost (bps) |
| `REBAL_EVERY` | `3` | Rebalance every N bars |
| `ENTRY_SCORE_MIN` | `5` | Min score to enter |
| `KF_PROC_NOISE` | `0.04` | Kalman process noise |
| `STOP_ATR_MULT` | `2.5` | ATR stop-loss multiplier |
| `KELLY_FRACTION` | `0.5` | Half-Kelly (1.0 = full Kelly) |

### `alphascout_ml.py`

| Parameter | Default | Description |
|---|---|---|
| `XGB_N_ESTIMATORS` | `500` | XGBoost tree count |
| `XGB_MIN_TRAIN` | `252` | Min bars before first XGB |
| `XGB_RETRAIN_EVERY` | `21` | Retrain every N bars |
| `XGB_LOOKAHEAD` | `5` | Forward return horizon |
| `LSTM_SEQ_LEN` | `60` | LSTM input window |
| `LSTM_HIDDEN` | `128` | LSTM hidden units |
| `LSTM_EPOCHS` | `80` | Max training epochs |
| `LSTM_LOOKAHEAD` | `20` | Regime label horizon |
| `AKF_PRETRAIN_BARS` | `300` | Adaptive KF training bars |

### `alphascout_live.py`

| Parameter | Default | Description |
|---|---|---|
| `MAX_POSITIONS` | `5` | Max concurrent positions |
| `MAX_POS_FRAC` | `0.20` | Max fraction per stock (tighter live) |
| `MIN_EQUITY_USD` | `10,000` | Hard halt threshold |
| `MAX_PORT_DD` | `0.12` | Halt if portfolio DD > 12% |
| `ML_ENTRY_SCORE` | `7` | ML-enhanced min score |
| `HIST_BARS` | `350` | Bars fetched per ticker |
| `DELAY_GUARD` | `False` | Skip today's bar |
| `DATA_FEED` | `iex` | `iex` = free, `sip` = paid |
| `RUN_TIME_ET` | `16:15` | Daily execution time |

---

## Risk Disclosures

- **Backtest ≠ live performance.** Transaction costs, slippage, and market
  impact are estimated; real costs vary.
- **Regime filter is not perfect.** Bear markets can begin abruptly.
- **ML models can overfit.** Walk-forward design minimises this but does
  not eliminate it.
- **Start with paper trading.** Use `--live` only after extended paper
  validation (suggest ≥ 3 months).
- **This is not financial advice.**
