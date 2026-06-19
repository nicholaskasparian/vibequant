#!/usr/bin/env python3
"""
AlphaScout Backtest  ·  Kalman–Kelly–OU Momentum Strategy
══════════════════════════════════════════════════════════

MATHEMATICAL FRAMEWORK
──────────────────────
1. KALMAN FILTER (2-state: level + velocity)
   ─────────────────────────────────────────
   State:  x  = [level, velocity]ᵀ
   Trans:  F  = [[1, dt], [0, 1]]          (constant-velocity model)
   Obs:    H  = [[1, 0]]                   (we only observe price level)
   Noise:  Q  = diag([q_L, q_V]),  R = scalar

   Predict:  x⁻ = Fx,       P⁻ = FPFᵀ + Q
   Innovate: ν  = y − Hx⁻,  S  = HP⁻Hᵀ + R
   Gain:     K  = P⁻Hᵀ / S
   Update:   x  = x⁻ + Kν,  P  = (I − KH)P⁻

   Applied on log-prices. Velocity > 0 ↔ uptrend (log-return scale).
   Ratio R/q_V sets speed/smoothness tradeoff: higher → smoother.

2. ORNSTEIN-UHLENBECK MEAN REVERSION
   ───────────────────────────────────
   dXₜ = θ(μ − Xₜ)dt + σ dWₜ

   Discrete: Xₜ₊₁ = a + b·Xₜ + εₜ,  εₜ ~ N(0, σ²·dt)
   OLS fit → θ̂ = (1 − b̂)/dt,  μ̂ = â/(1 − b̂),  σ̂ = std(ε)/√dt
   σ_eq = σ̂/√(2θ̂)                    (equilibrium std deviation)
   Half-life τ = ln(2)/θ̂              (mean-reversion speed)
   Z-score z  = (Xₜ − μ̂) / σ_eq     (standardised deviation)

   Applied to residuals (log-price − Kalman level). Low z → entry.

3. HALF-KELLY CRITERION
   ──────────────────────
   For log-normal returns: optimal fraction f* = μ/σ² (annualised).
   Half-Kelly: f = f*/2 — near-optimal expected log-growth, half variance.
   Rolling μ̂ and σ̂² on 60-day window. Position capped at MAX_POS_FRAC.

4. SHARPE MOMENTUM  (cross-sectional ranking)
   ─────────────────────────────────────────
   SRᵢ = (μ̂ᵢ · 252) / (σ̂ᵢ · √252)   over 20-day rolling window
   Penalises high-vol stocks; selects risk-efficient momentum.
   Used to rank stocks after score threshold is met.

5. EWMA-GARCH VOLATILITY REGIME SCALING
   ──────────────────────────────────────
   σ²ₜ = λ·σ²ₜ₋₁ + (1−λ)·rₜ²,   λ = 0.94  (RiskMetrics)
   Regime ratio ρ = σ_EWMA / σ_longrun:
     ρ ≥ 2.0  → scalar 0.50  (high-vol, cut size 50%)
     ρ ≥ 1.5  → scalar 0.75
     ρ < 0.75 → scalar 1.25  (quiet period, add 25%)
     else     → scalar 1.00

DATA-DELAY MODEL  (for free data streams)
──────────────────────────────────────────
Signal computed on bar t executes at bar t + DATA_DELAY_BARS.
Simulated by shifting the price lookup forward by that many indices.
  • Free EOD (yfinance/Alpaca basic) → DATA_DELAY_BARS = 1 (recommended)
  • Extra caution / intraday free     → DATA_DELAY_BARS = 2
Execution costs include commission + half-spread + market impact.

ON "1% DAILY" RETURNS:
  1%/day = (1.01²⁵² − 1) ≈ 1,147% annual.  No strategy sustains this.
  Renaissance Medallion (best ever): ~66% net/yr.  SPY: ~10% long-run.
  This strategy targets 25-50% CAGR, Sharpe > 1.5 — top-1% globally.
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('AlphaScout')

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BACKTEST_START  = '2020-01-01'
BACKTEST_END    = '2025-05-30'
WARMUP_START    = '2019-01-01'   # extra history for indicator burn-in

INITIAL_CAPITAL = 100_000.0
MAX_POSITIONS   = 5
MAX_POS_FRAC    = 0.25           # max fraction of equity per name

# ── Data delay & execution cost model ─────────────────────────────────────────
DATA_DELAY_BARS   = 1            # 1 = next bar (standard EOD), 2 = extra conservative
COMMISSION_BPS    = 10           # 0.10 % per transaction (Alpaca paper)
SPREAD_BPS        = 4            # 0.04 % half-spread estimate
IMPACT_BPS        = 2            # 0.02 % market-impact estimate
COST_BPS          = COMMISSION_BPS + SPREAD_BPS + IMPACT_BPS  # 16 bps total per side

# ── Rebalancing ────────────────────────────────────────────────────────────────
REBAL_EVERY     = 3              # full rebalance every N bars (reduces turnover)
MIN_HOLD_BARS   = 2              # minimum hold before allowing exit
ENTRY_SCORE_MIN = 5              # composite score (0–9) required to enter
EXIT_SCORE_MAX  = 2              # exit if score falls to this level

# ── Kalman filter ─────────────────────────────────────────────────────────────
KF_OBS_NOISE   = 1.0
KF_PROC_NOISE  = 0.04            # higher → adapts faster (less smooth)

# ── OU process ────────────────────────────────────────────────────────────────
OU_FIT_WINDOW  = 60
OU_ENTRY_Z     = -1.5            # entry on dip: z < -1.5 = oversold vs equilibrium
OU_EXIT_Z      =  1.0            # mean-reversion trade complete at z > 1.0

# ── Kelly ─────────────────────────────────────────────────────────────────────
KELLY_WINDOW   = 60
KELLY_FRACTION = 0.5             # half-Kelly

# ── Volatility regime ─────────────────────────────────────────────────────────
EWMA_LAMBDA    = 0.94            # RiskMetrics daily decay factor
VOL_LT_WINDOW  = 252
ATR_PERIOD     = 14
STOP_ATR_MULT  = 2.5

# ── Universe ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD',
    'AVGO', 'NFLX', 'JPM',  'V',    'UNH',  'LLY',   'MELI', 'PLTR',
    'SHOP', 'MSTR', 'COIN', 'SOFI',
]
BENCHMARK = 'SPY'


# ══════════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class KalmanFilter:
    """
    Two-state constant-velocity Kalman filter on log-prices.

    State:        x = [level, velocity]ᵀ
    Transition:   F = [[1, dt], [0, 1]]
    Observation:  H = [[1, 0]]
    Process cov:  Q = diag([q_L, q_V])
    Obs noise:    R (scalar)

    The velocity estimate is the key signal: positive → uptrend,
    negative → downtrend.  Smoother than any MA because the Kalman
    gain is computed optimally from the noise ratio R/Q.
    """

    def __init__(self,
                 obs_noise: float  = KF_OBS_NOISE,
                 proc_noise: float = KF_PROC_NOISE,
                 dt: float         = 1.0):
        self.R  = obs_noise
        self.q_L = proc_noise * 0.1       # level  noise (slower)
        self.q_V = proc_noise              # velocity noise (faster)
        self.dt  = dt
        self.F   = np.array([[1.0, dt ], [0.0, 1.0]])
        self.H   = np.array([[1.0, 0.0]])
        self.Q   = np.diag([self.q_L, self.q_V])
        self._reset()

    def _reset(self):
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1e4          # diffuse initialisation

    def step(self, y: float) -> Tuple[float, float, float]:
        """One measurement update.  Returns (level, velocity, innovation)."""
        # Predict
        x_p = self.F @ self.x
        P_p = self.F @ self.P @ self.F.T + self.Q
        # Innovation
        innov = y - (self.H @ x_p)[0]
        S     = (self.H @ P_p @ self.H.T)[0, 0] + self.R
        # Kalman gain
        K     = (P_p @ self.H.T)[:, 0] / S
        # Update
        self.x = x_p + K * innov
        self.P = (np.eye(2) - np.outer(K, self.H[0])) @ P_p
        return self.x[0], self.x[1], innov

    def filter_series(self, log_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch filter.  Returns (levels, velocities, innovations)."""
        self._reset()
        n = len(log_prices)
        lv, vl, iv = np.empty(n), np.empty(n), np.empty(n)
        for i, y in enumerate(log_prices):
            lv[i], vl[i], iv[i] = self.step(float(y))
        return lv, vl, iv


class OUProcess:
    """
    Ornstein-Uhlenbeck parameter estimator via OLS.

    Fits:  Xₜ₊₁ = a + b·Xₜ + εₜ
    Then:  θ̂ = (1−b̂)/dt,  μ̂ = â/(1−b̂),  σ̂ = std(ε)/√dt
    σ_eq  = σ̂/√(2θ̂)   (std dev at equilibrium)
    τ     = ln(2)/θ̂    (half-life in bars)
    z     = (Xₜ−μ̂)/σ_eq

    Applied to residuals (log-price − Kalman level) to find statistically
    meaningful oversold / overbought departures from the Kalman trend.
    """

    @staticmethod
    def fit(series: np.ndarray, dt: float = 1.0) -> Dict:
        x, y  = series[:-1], series[1:]
        b, a  = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        theta = max((1.0 - b) / dt, 1e-6)
        mu    = a / (1.0 - b) if abs(1.0 - b) > 1e-9 else float(np.mean(series))
        sigma = float(np.std(resid)) / np.sqrt(dt)
        sigma_eq  = sigma / np.sqrt(max(2.0 * theta, 1e-9))
        half_life = np.log(2.0) / theta
        zscore    = (series[-1] - mu) / (sigma_eq + 1e-9)
        return dict(theta=theta, mu=mu, sigma=sigma,
                    sigma_eq=sigma_eq, half_life=half_life, zscore=zscore)

    @staticmethod
    def rolling_zscore(residuals: pd.Series, window: int = OU_FIT_WINDOW) -> pd.Series:
        """Vectorised rolling OU z-score (one OLS fit per bar)."""
        arr = residuals.values
        zs  = np.full(len(arr), np.nan)
        for i in range(window, len(arr)):
            try:
                zs[i] = OUProcess.fit(arr[i - window: i + 1])['zscore']
            except Exception:
                pass
        return pd.Series(zs, index=residuals.index, name='ou_z')


class HalfKelly:
    """
    Continuous Half-Kelly fraction  f = (μ / σ²) / 2

    Maximises E[log W] with half the variance of full Kelly.
    Proven asymptotically optimal for long-run geometric wealth growth
    while dramatically reducing probability of ruin.

    Rolling window estimator ensures adaptation to changing dynamics.
    """

    def __init__(self,
                 window:   int   = KELLY_WINDOW,
                 fraction: float = KELLY_FRACTION,
                 max_frac: float = MAX_POS_FRAC):
        self.window   = window
        self.fraction = fraction
        self.max_frac = max_frac

    def fractions(self, returns: pd.Series) -> pd.Series:
        mu  = returns.rolling(self.window).mean() * 252
        var = returns.rolling(self.window).var()  * 252
        f   = self.fraction * mu / (var.clip(lower=1e-9))
        return f.clip(0.0, self.max_frac)


class EWMAVol:
    """
    EWMA volatility model  σ²ₜ = λ·σ²ₜ₋₁ + (1−λ)·rₜ²

    λ = 0.94 is the RiskMetrics daily parameter (JP Morgan, 1994).
    Regime scalar multiplies the Kelly/equal-weight position size
    to reduce exposure during volatility spikes.
    """

    def __init__(self,
                 lam: float      = EWMA_LAMBDA,
                 lt_window: int  = VOL_LT_WINDOW):
        self.lam      = lam
        self.lt_window = lt_window

    def compute(self, returns: pd.Series) -> pd.DataFrame:
        # EWMA span corresponding to given λ: span = 2/(1−λ) − 1
        span    = int(2.0 / (1.0 - self.lam) - 1)
        vol_ewm = returns.ewm(span=span).std() * np.sqrt(252)
        vol_lt  = returns.rolling(self.lt_window).std() * np.sqrt(252)
        ratio   = vol_ewm / (vol_lt + 1e-9)
        scalar  = np.where(ratio >= 2.0, 0.50,
                  np.where(ratio >= 1.5, 0.75,
                  np.where(ratio < 0.75, 1.25, 1.00))).astype(float)
        return pd.DataFrame({
            'vol_ewm'   : vol_ewm,
            'vol_lt'    : vol_lt,
            'vol_ratio' : ratio,
            'vol_scalar': scalar,
        }, index=returns.index)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

def download_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Download adjusted OHLCV via yfinance, return per-ticker DataFrames."""
    all_t = sorted(set(tickers + [BENCHMARK]))
    log.info(f'Downloading {len(all_t)} tickers  {WARMUP_START} → {BACKTEST_END}')
    raw = yf.download(all_t, start=WARMUP_START, end=BACKTEST_END,
                      auto_adjust=True, progress=False, threads=True)
    out: Dict[str, pd.DataFrame] = {}
    for t in all_t:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = pd.DataFrame({
                    'open'  : raw['Open'  ][t],
                    'high'  : raw['High'  ][t],
                    'low'   : raw['Low'   ][t],
                    'close' : raw['Close' ][t],
                    'volume': raw['Volume'][t],
                }).dropna()
            else:
                df = raw[['Open','High','Low','Close','Volume']].copy()
                df.columns = ['open','high','low','close','volume']
                df.dropna(inplace=True)
            df.index = pd.to_datetime(df.index)
            out[t] = df
            log.info(f'  {t:6s}: {len(df)} bars')
        except Exception as e:
            log.warning(f'  {t}: skip ({e})')
    return out


def compute_features(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Run all mathematical models on each ticker."""
    kf    = KalmanFilter()
    ewma  = EWMAVol()
    kelly = HalfKelly()
    out: Dict[str, pd.DataFrame] = {}

    for t, df in data.items():
        d = df.copy()
        c = d['close']
        r = c.pct_change()

        # 1. Kalman filter on log-prices ──────────────────────────────────
        lp       = np.log(c.values.astype(float))
        kf._reset()
        lv, vel, innov = kf.filter_series(lp)
        d['kf_level']  = lv
        d['kf_vel']    = vel              # log-return velocity
        d['kf_price']  = np.exp(lv)      # back to price space
        d['kf_innov']  = innov            # raw innovations (useful for diagnostics)

        # 2. OU residuals → z-score ───────────────────────────────────────
        residual = pd.Series(lp - lv, index=c.index, name='resid')
        d['ou_z'] = OUProcess.rolling_zscore(residual, window=OU_FIT_WINDOW)

        # 3. ATR (Wilder / EWM) ───────────────────────────────────────────
        tr = pd.concat([
            d['high'] - d['low'],
            (d['high'] - c.shift()).abs(),
            (d['low']  - c.shift()).abs(),
        ], axis=1).max(axis=1)
        d['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

        # 4. EWMA-GARCH vol regime ─────────────────────────────────────────
        d = d.join(ewma.compute(r))

        # 5. Half-Kelly fraction ───────────────────────────────────────────
        d['kelly_f'] = kelly.fractions(r)

        # 6. Sharpe momentum (risk-adjusted cross-sectional rank) ──────────
        d['ret_20']   = c.pct_change(20)
        d['vol_20']   = r.rolling(20).std() * np.sqrt(252)
        d['sharpe_m'] = (d['ret_20'] * 252 / 20.0) / (d['vol_20'] + 1e-9)
        d['sharpe_m'] = d['sharpe_m'].clip(-3.0, 10.0)

        # 7. MACD cross (entry confirmation) ──────────────────────────────
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        d['macd']     = ema12 - ema26
        d['macd_sig'] = d['macd'].ewm(span=9, adjust=False).mean()

        # 8. Volume ratio ─────────────────────────────────────────────────
        d['vol_ratio'] = d['volume'] / d['volume'].rolling(20).mean()

        out[t] = d

    log.info('Feature engineering complete.')
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_row(row: pd.Series) -> int:
    """
    Composite signal score  ∈  {0, …, 9}.

    Kalman signals   (3 pts): upward velocity, price > KF level, regime OK
    OU signals       (2 pts): z-score in entry zone, not blown-off
    Momentum signals (2 pts): Sharpe mom > 0.5 and > 1.0
    MACD             (1 pt):  MACD line above signal
    Volume           (1 pt):  above-average volume
    """
    s = 0
    # Kalman: positive velocity (scaled for log-return units)
    if row.get('kf_vel',    0.0) > 2e-4:              s += 1
    # Kalman: close above smoothed KF price level
    if row.get('close',     0.0) > row.get('kf_price', 0.0): s += 1
    # Vol regime: not in crisis (scalar not at minimum)
    if row.get('vol_scalar', 1.0) >= 0.75:             s += 1
    # OU: z-score not too high (avoid blow-off top entries)
    ou_z = float(row.get('ou_z', 0.0))
    if -3.5 < ou_z < 1.5:                              s += 1
    # OU: either mean-reversion entry OR uptrend continuation
    if ou_z < OU_ENTRY_Z or 0.0 < ou_z < OU_EXIT_Z:   s += 1
    # Sharpe momentum – moderate
    if row.get('sharpe_m', 0.0) > 0.5:                 s += 1
    # Sharpe momentum – strong
    if row.get('sharpe_m', 0.0) > 1.0:                 s += 1
    # MACD cross
    if row.get('macd', 0.0) > row.get('macd_sig', 0.0): s += 1
    # Volume confirmation
    if row.get('vol_ratio', 1.0) > 1.1:                s += 1
    return s


def market_regime(spy_feat: pd.DataFrame, date: pd.Timestamp) -> float:
    """
    SPY market regime based on smoothed Kalman velocity.

    Uses a 10-bar window average of KF velocity to avoid whipsaws.
    Returns:  0.0 → full cash (bear)
              0.5 → half exposure (mild downtrend)
              1.0 → full exposure (bull)
    """
    try:
        past     = spy_feat.loc[:date].tail(10)
        vel_mean = float(past['kf_vel'].mean())
        if vel_mean < -1e-3:  return 0.0
        if vel_mean < 0.0:    return 0.5
        return 1.0
    except Exception:
        return 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Event-driven daily backtester.

    Data-delay model
    ─────────────────
    Signal computed at bar t's close.
    Execution price looked up at bar t + DATA_DELAY_BARS.
    Entry/exit costs: COST_BPS applied directionally (buy = add, sell = subtract).

    Position management
    ────────────────────
    Positions stored as: {ticker: {shares, entry_px, stop_px, entry_bar, score}}
    Stop losses checked every bar (no delay — assumes GTC-stop order is live).
    Full rebalance every REBAL_EVERY bars.  MIN_HOLD_BARS prevents immediate flip.
    """

    def __init__(self, features: Dict[str, pd.DataFrame]):
        self.feat   = features
        self.spy    = features[BENCHMARK]
        self.dates  = self.spy.loc[BACKTEST_START:BACKTEST_END].index
        self.cash   = INITIAL_CAPITAL
        self.pos: Dict[str, Dict]  = {}
        self.equity_log: List      = []
        self.trade_log:  List      = []
        self.bar        = 0

    # ── price helpers ─────────────────────────────────────────────────────

    def _exec_price(self, t: str, date: pd.Timestamp, side: str) -> Optional[float]:
        """
        Simulated fill price: close[t + DATA_DELAY_BARS] ± costs.
        Falls back to current bar if future bar unavailable.
        """
        try:
            f   = self.feat[t]
            i   = f.index.get_loc(date)
            j   = min(i + DATA_DELAY_BARS, len(f) - 1)
            px  = float(f.iloc[j]['close'])
            bps = COST_BPS / 10_000
            return px * (1 + bps) if side == 'buy' else px * (1 - bps)
        except Exception:
            return None

    def _spot(self, t: str, date: pd.Timestamp) -> Optional[float]:
        try:    return float(self.feat[t].loc[date, 'close'])
        except: return None

    def _equity(self, date: pd.Timestamp) -> float:
        v = self.cash
        for t, p in self.pos.items():
            px = self._spot(t, date)
            if px: v += p['shares'] * px
        return v

    # ── trade execution ────────────────────────────────────────────────────

    def _sell(self, t: str, date: pd.Timestamp, reason: str):
        p = self.pos.pop(t, None)
        if not p: return
        px = self._exec_price(t, date, 'sell') or p['entry_px']
        self.cash += p['shares'] * px
        self.trade_log.append(dict(date=date, ticker=t, side='SELL',
            reason=reason, price=px, shares=p['shares'],
            pnl=p['shares'] * (px - p['entry_px'])))

    def _buy(self, t: str, date: pd.Timestamp, alloc_usd: float, score: int, atr: float):
        px = self._exec_price(t, date, 'buy')
        if not px or px <= 0: return
        shares = int(alloc_usd / px)
        cost   = shares * px
        if shares <= 0 or cost > self.cash: return
        self.cash -= cost
        self.pos[t] = dict(shares=shares, entry_px=px,
                           stop_px=px - STOP_ATR_MULT * atr,
                           entry_bar=self.bar, score=score)
        self.trade_log.append(dict(date=date, ticker=t, side='BUY',
            reason='signal', price=px, shares=shares, pnl=0))

    # ── main loop ──────────────────────────────────────────────────────────

    def run(self) -> pd.Series:
        log.info(f'Running backtest  {BACKTEST_START} → {BACKTEST_END}  '
                 f'| delay={DATA_DELAY_BARS}bar | cost={COST_BPS}bps/side')
        for i, date in enumerate(self.dates):
            self.bar = i

            # 1. Stop-loss check ──────────────────────────────────────────
            for t in list(self.pos.keys()):
                px = self._spot(t, date)
                if px and px <= self.pos[t]['stop_px']:
                    self._sell(t, date, 'stop_loss')

            # 2. Market regime ────────────────────────────────────────────
            regime = market_regime(self.spy, date)
            if regime == 0.0:
                for t in list(self.pos.keys()):
                    self._sell(t, date, 'regime_off')
                self.equity_log.append((date, self.cash))
                continue

            # 3. Score universe ───────────────────────────────────────────
            scores: Dict[str, int]       = {}
            rows:   Dict[str, pd.Series] = {}
            for t in UNIVERSE:
                f = self.feat.get(t)
                if f is None or date not in f.index:
                    continue
                row = f.loc[date]
                if row.isnull().any():
                    continue
                sc = score_row(row)
                scores[t] = sc
                rows[t]   = row

            # 4. Signal exits (only on rebalance bars + min hold) ─────────
            if i % REBAL_EVERY == 0:
                for t in list(self.pos.keys()):
                    held = i - self.pos[t]['entry_bar']
                    if held >= MIN_HOLD_BARS and scores.get(t, 0) <= EXIT_SCORE_MAX:
                        self._sell(t, date, 'score_exit')

            # 5. Target portfolio (rank qualified stocks by Sharpe mom) ───
            qualifiers = [(t, rows[t].get('sharpe_m', 0.0))
                          for t, sc in scores.items() if sc >= ENTRY_SCORE_MIN]
            qualifiers.sort(key=lambda x: -x[1])
            target = [t for t, _ in qualifiers[:MAX_POSITIONS]]

            # 6. Rebalance: exit stale positions ──────────────────────────
            if i % REBAL_EVERY == 0:
                for t in list(self.pos.keys()):
                    if t not in target:
                        held = i - self.pos[t]['entry_bar']
                        if held >= MIN_HOLD_BARS:
                            self._sell(t, date, 'rebal_exit')

            # 7. Total equity after exits ─────────────────────────────────
            total = self._equity(date)

            # 8. Enter new positions (Kelly-sized, vol-regime scaled) ──────
            for t in target:
                if t in self.pos:
                    continue
                row        = rows.get(t)
                if row is None: continue
                kelly_f    = float(row.get('kelly_f',    MAX_POS_FRAC / MAX_POSITIONS))
                vol_scl    = float(row.get('vol_scalar', 1.0))
                frac       = min(kelly_f * vol_scl * regime, MAX_POS_FRAC)
                frac       = max(frac, 0.05)
                atr        = float(row.get('atr', row['close'] * 0.02))
                self._buy(t, date, total * frac, scores[t], atr)

            # 9. Record equity ────────────────────────────────────────────
            self.equity_log.append((date, self._equity(date)))

        eq = pd.Series(
            [v for _, v in self.equity_log],
            index=pd.DatetimeIndex([d for d, _ in self.equity_log]),
            name='AlphaScout',
        )
        log.info(f'Backtest complete — {len(self.trade_log)} trades')
        return eq


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def metrics(eq: pd.Series, label: str = 'Strategy') -> Dict:
    r      = eq.pct_change().dropna()
    years  = (eq.index[-1] - eq.index[0]).days / 365.25
    total  = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr   = (1.0 + total) ** (1.0 / years) - 1.0
    ann_vol= r.std() * np.sqrt(252)
    sharpe = r.mean() * 252 / (ann_vol + 1e-9)
    down_v = r[r < 0].std() * np.sqrt(252)
    sortino= r.mean() * 252 / (down_v + 1e-9)
    dd     = (eq - eq.cummax()) / eq.cummax()
    max_dd = dd.min()
    calmar = cagr / (-max_dd + 1e-9)
    mo_ret = eq.resample('ME').last().pct_change().dropna()
    return {
        'Label'              : label,
        'Total Return'       : f'{total*100:+.1f}%',
        'CAGR'               : f'{cagr*100:.1f}%',
        'Avg Daily Return'   : f'{r.mean()*100:.3f}%',
        'Annual Volatility'  : f'{ann_vol*100:.1f}%',
        'Sharpe Ratio'       : f'{sharpe:.2f}',
        'Sortino Ratio'      : f'{sortino:.2f}',
        'Max Drawdown'       : f'{max_dd*100:.1f}%',
        'Calmar Ratio'       : f'{calmar:.2f}',
        'Win Rate (daily)'   : f'{(r>0).mean()*100:.1f}%',
        'Win Rate (monthly)' : f'{(mo_ret>0).mean()*100:.1f}%',
        'Best Month'         : f'{mo_ret.max()*100:.1f}%',
        'Worst Month'        : f'{mo_ret.min()*100:.1f}%',
        # raw floats for comparison logic
        '_cagr'    : cagr,
        '_sharpe'  : sharpe,
        '_max_dd'  : max_dd,
        '_sortino' : sortino,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(strat: pd.Series, bench: pd.Series,
             sm: Dict, bm: Dict, trades: pd.DataFrame,
             path: str = '/mnt/user-data/outputs/alphascout_backtest.png'):

    common = strat.index.intersection(bench.index)
    s, b   = strat.loc[common], bench.loc[common]
    sn     = s / s.iloc[0] * 100
    bn     = b / b.iloc[0] * 100

    BG, P, G2 = '#0d1117', '#161b22', '#21262d'
    BLUE, GRN, RED, GREY, TEXT = '#58a6ff', '#3fb950', '#f85149', '#8b949e', '#e6edf3'

    fig = plt.figure(figsize=(20, 15), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            height_ratios=[2.5, 1.5, 1.5], hspace=0.42, wspace=0.32)

    def _ax(sp, title=''):
        a = fig.add_subplot(sp)
        a.set_facecolor(P); a.spines[:].set_color(G2)
        a.tick_params(colors=GREY, labelsize=8)
        a.yaxis.label.set_color(GREY); a.xaxis.label.set_color(GREY)
        a.grid(True, color=G2, lw=0.5)
        if title: a.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=7)
        return a

    # ── Equity curve ────────────────────────────────────────────────────
    ax1 = _ax(gs[0, :], 'Portfolio Equity  (Normalised to 100)')
    ax1.plot(sn.index, sn.values, color=BLUE, lw=2.0, label='AlphaScout')
    ax1.plot(bn.index, bn.values, color=GREY, lw=1.5, ls='--', label='SPY Buy & Hold')
    ax1.fill_between(sn.index, sn.values, bn.values,
                     where=sn.values >= bn.values, alpha=0.12, color=BLUE)
    ax1.fill_between(sn.index, sn.values, bn.values,
                     where=sn.values <  bn.values, alpha=0.12, color=RED)
    ax1.legend(facecolor=P, edgecolor=G2, labelcolor=TEXT, fontsize=9)
    ax1.set_ylabel('Normalised Value', color=GREY, fontsize=9)

    # ── Drawdown ────────────────────────────────────────────────────────
    ax2 = _ax(gs[1, :2], 'Strategy Drawdown (%)')
    dd  = (s - s.cummax()) / s.cummax() * 100
    ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.5)
    ax2.plot(dd.index, dd.values, color=RED, lw=0.8)
    ax2.set_ylabel('%', color=GREY, fontsize=9)

    # ── Rolling 60-day Sharpe ────────────────────────────────────────────
    ax3 = _ax(gs[1, 2], '60-Day Rolling Sharpe')
    rs  = s.pct_change().dropna()
    rsr = rs.rolling(60).apply(
        lambda x: x.mean() / (x.std() + 1e-9) * np.sqrt(252), raw=True)
    ax3.plot(rsr.index, rsr.values, color=GRN, lw=1.3)
    for y, lbl, col in [(0,'SR=0',GREY),(1,'SR=1',BLUE),(2,'SR=2',GRN)]:
        ax3.axhline(y, color=col, lw=0.8, ls='--', label=lbl)
    ax3.legend(facecolor=P, edgecolor=G2, labelcolor=TEXT, fontsize=7)
    ax3.set_ylabel('Sharpe', color=GREY, fontsize=9)

    # ── Monthly returns heatmap ──────────────────────────────────────────
    ax4  = _ax(gs[2, :2], 'Monthly Returns (%)')
    mo   = s.resample('ME').last().pct_change().dropna() * 100
    mdf  = pd.DataFrame({'y': mo.index.year, 'm': mo.index.month, 'r': mo.values})
    pvt  = mdf.pivot(index='y', columns='m', values='r')
    mlab = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    vmax = max(abs(pvt.values[~np.isnan(pvt.values)]).max(), 5.0)
    cmap = LinearSegmentedColormap.from_list('rg', [RED, '#ffffff', GRN])
    im   = ax4.imshow(pvt.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax4.set_xticks(range(pvt.shape[1]))
    ax4.set_xticklabels([mlab[c-1] for c in pvt.columns], fontsize=7, color=GREY)
    ax4.set_yticks(range(pvt.shape[0]))
    ax4.set_yticklabels(pvt.index, fontsize=7, color=GREY)
    for i in range(pvt.shape[0]):
        for j in range(pvt.shape[1]):
            v = pvt.values[i, j]
            if not np.isnan(v):
                ax4.text(j, i, f'{v:.1f}', ha='center', va='center',
                         fontsize=6.2, color='#000' if abs(v) < vmax*0.55 else '#fff')
    plt.colorbar(im, ax=ax4, fraction=0.025, pad=0.01)

    # ── Metrics table ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor(P); ax5.axis('off')
    rows_tb = []
    for m in ['Total Return','CAGR','Avg Daily Return','Annual Volatility',
              'Sharpe Ratio','Sortino Ratio','Max Drawdown',
              'Calmar Ratio','Win Rate (monthly)','Best Month','Worst Month']:
        rows_tb.append([m, sm.get(m,'—'), bm.get(m,'—')])
    tbl = ax5.table(cellText=rows_tb, colLabels=['Metric','AlphaScout','SPY'],
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(BG if r == 0 else P)
        cell.set_text_props(color=(BLUE if c == 1 and r > 0 else
                                   GRN  if r == 0 else TEXT))
        cell.set_edgecolor(G2)
    ax5.set_title('Performance Summary', color=TEXT, fontsize=11,
                  fontweight='bold', pad=7)

    fig.suptitle(
        'AlphaScout  ·  Kalman–Kelly–OU Momentum  ·  Backtest Results',
        fontsize=14, fontweight='bold', color=TEXT, y=1.01,
    )
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    log.info(f'Chart saved → {path}')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info('═' * 62)
    log.info(' AlphaScout — Backtest')
    log.info(f'  Period  : {BACKTEST_START} → {BACKTEST_END}')
    log.info(f'  Capital : ${INITIAL_CAPITAL:,.0f}')
    log.info(f'  Universe: {len(UNIVERSE)} stocks   Max positions: {MAX_POSITIONS}')
    log.info(f'  Delay   : {DATA_DELAY_BARS} bar(s)   Cost: {COST_BPS} bps/side')
    log.info('═' * 62)

    raw      = download_data(UNIVERSE)
    features = compute_features(raw)

    # Benchmark
    spy_px  = features[BENCHMARK].loc[BACKTEST_START:BACKTEST_END, 'close']
    bench   = spy_px / spy_px.iloc[0] * INITIAL_CAPITAL
    bench.name = 'SPY'

    # Strategy
    engine  = BacktestEngine(features)
    strat   = engine.run()

    # Metrics
    sm = metrics(strat, 'AlphaScout')
    bm = metrics(bench, 'SPY B&H')

    # Print
    keys = ['Total Return','CAGR','Avg Daily Return','Annual Volatility',
            'Sharpe Ratio','Sortino Ratio','Max Drawdown',
            'Calmar Ratio','Win Rate (daily)','Win Rate (monthly)']
    w = 22
    print(f'\n{"═"*60}')
    print(f'  {"METRIC":<{w}}  {"AlphaScout":>14}  {"SPY B&H":>12}')
    print(f'{"─"*60}')
    for k in keys:
        print(f'  {k:<{w}}  {sm[k]:>14}  {bm[k]:>12}')
    print(f'{"═"*60}')
    print()

    beat_cagr   = sm['_cagr']    > bm['_cagr']
    beat_sharpe = sm['_sharpe']  > bm['_sharpe']
    beat_dd     = sm['_max_dd']  > bm['_max_dd']
    beat_sort   = sm['_sortino'] > bm['_sortino']

    labels = [
        ('CAGR',          beat_cagr),
        ('Sharpe Ratio',  beat_sharpe),
        ('Max Drawdown',  beat_dd),
        ('Sortino Ratio', beat_sort),
    ]
    for lbl, ok in labels:
        print(f'  {"✓" if ok else "✗"}  Beats SPY on {lbl}')
    print()

    trades_df = pd.DataFrame(engine.trade_log)
    if not trades_df.empty:
        print(f'  Total trades   : {len(trades_df)}')
        wins = trades_df[trades_df["side"]=="SELL"]["pnl"]
        if len(wins):
            print(f'  Trade win rate : {(wins > 0).mean()*100:.1f}%')
            print(f'  Avg trade P&L  : ${wins.mean():+.0f}')
    print()

    plot_all(strat, bench, sm, bm, trades_df)


def main_ml():
    """
    ML-enhanced backtest.  Identical to main() but:
      • Augments features with XGBoost signal + LSTM regime probability
      • Uses score_row_ml() (0–12 scale) instead of score_row() (0–9)
      • Replaces Kalman-velocity regime filter with LSTM P(bull)
      • Raises ENTRY_SCORE_MIN to 7 to compensate for wider score range
    Requires alphascout_ml.py in the same directory.
    """
    try:
        from alphascout_ml import (
            MLPipeline, build_ml_features,
            score_row_ml, regime_from_ml,
        )
    except ImportError:
        log.error('alphascout_ml.py not found — run standard main() instead')
        return

    log.info('═' * 62)
    log.info(' AlphaScout — ML-Enhanced Backtest')
    log.info(f'  Period  : {BACKTEST_START} → {BACKTEST_END}')
    log.info(f'  Capital : ${INITIAL_CAPITAL:,.0f}')
    log.info(f'  Universe: {len(UNIVERSE)} stocks   Max positions: {MAX_POSITIONS}')
    log.info(f'  Delay   : {DATA_DELAY_BARS} bar(s)   Cost: {COST_BPS} bps/side')
    log.info('═' * 62)

    raw      = download_data(UNIVERSE)
    features = compute_features(raw)

    # ── Train / load ML pipeline ─────────────────────────────────────────
    import os
    MODEL_DIR = './alphascout_models'
    pipeline  = MLPipeline(use_akf=False)
    lstm_path = os.path.join(MODEL_DIR, 'lstm_regime.pt')

    if os.path.exists(lstm_path):
        log.info(f'Loading ML models from {MODEL_DIR}/')
        pipeline.load(MODEL_DIR, tickers=UNIVERSE, benchmark=BENCHMARK)
        ml_features = build_ml_features(features)
        # Rebuild XGB walk-forward predictions (they weren't saved per-bar)
        from alphascout_ml import WalkForwardXGB, LSTM_FEATURE_COLS
        spy_df = ml_features.get(BENCHMARK)
        for t in UNIVERSE:
            df = ml_features.get(t)
            if df is None: continue
            xgb_m = pipeline.xgb_models.get(t)
            if xgb_m and xgb_m._model is not None:
                df['xgb_prob'] = xgb_m.fit_predict(df)
            else:
                xgb_m = WalkForwardXGB()
                df['xgb_prob'] = xgb_m.fit_predict(df)
                pipeline.xgb_models[t] = xgb_m
            if spy_df is not None:
                df['regime_prob'] = pipeline.lstm.predict_series(spy_df).reindex(df.index).ffill()
            ml_features[t] = df
    else:
        log.info('Training ML pipeline from scratch (this takes a few minutes) …')
        ml_features = pipeline.augment(features)
        os.makedirs(MODEL_DIR, exist_ok=True)
        pipeline.save(MODEL_DIR)

    # ── Patch the engine to use ML scoring + LSTM regime ─────────────────
    # We monkey-patch the two methods that consume signals.
    import types

    def _score(row: pd.Series) -> int:
        return score_row_ml(row, pipeline)

    def _regime(spy_feat: pd.DataFrame, date: pd.Timestamp) -> float:
        try:
            window = spy_feat.loc[:date].tail(LSTM_SEQ_LEN + 10)
        except Exception:
            window = spy_feat
        prob = pipeline.lstm.predict_prob(window)
        return pipeline.lstm.regime_scalar(prob)

    LSTM_SEQ_LEN = 60   # matches LSTM_SEQ_LEN in alphascout_ml

    # Benchmark
    spy_px  = ml_features[BENCHMARK].loc[BACKTEST_START:BACKTEST_END, 'close']
    bench   = spy_px / spy_px.iloc[0] * INITIAL_CAPITAL
    bench.name = 'SPY'

    # Temporarily override module-level functions used inside BacktestEngine
    import alphascout_backtest as _bt
    _orig_score  = _bt.score_row
    _orig_regime = _bt.market_regime
    _bt.score_row     = _score
    _bt.market_regime = _regime
    # Tighten entry threshold for wider 0–12 scale
    _bt.ENTRY_SCORE_MIN = 7
    _bt.EXIT_SCORE_MAX  = 3

    engine  = BacktestEngine(ml_features)
    strat   = engine.run()

    # Restore
    _bt.score_row      = _orig_score
    _bt.market_regime  = _orig_regime
    _bt.ENTRY_SCORE_MIN = ENTRY_SCORE_MIN
    _bt.EXIT_SCORE_MAX  = EXIT_SCORE_MAX

    sm = metrics(strat, 'AlphaScout+ML')
    bm = metrics(bench, 'SPY B&H')

    keys = ['Total Return','CAGR','Avg Daily Return','Annual Volatility',
            'Sharpe Ratio','Sortino Ratio','Max Drawdown',
            'Calmar Ratio','Win Rate (daily)','Win Rate (monthly)']
    w = 22
    print(f'\n{"═"*62}')
    print(f'  {"METRIC":<{w}}  {"AlphaScout+ML":>16}  {"SPY B&H":>12}')
    print(f'{"─"*62}')
    for k in keys:
        print(f'  {k:<{w}}  {sm[k]:>16}  {bm[k]:>12}')
    print(f'{"═"*62}')

    beat_cagr   = sm['_cagr']    > bm['_cagr']
    beat_sharpe = sm['_sharpe']  > bm['_sharpe']
    beat_dd     = sm['_max_dd']  > bm['_max_dd']
    print(f'\n  {"✓" if beat_cagr   else "✗"}  Beats SPY on CAGR')
    print(f'  {"✓" if beat_sharpe else "✗"}  Beats SPY on Sharpe')
    print(f'  {"✓" if beat_dd     else "✗"}  Lower max drawdown\n')

    trades_df = pd.DataFrame(engine.trade_log)
    plot_all(strat, bench, sm, bm, trades_df,
             path='/mnt/user-data/outputs/alphascout_ml_backtest.png')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='AlphaScout Backtest')
    ap.add_argument('--ml', action='store_true',
                    help='Use ML-enhanced signals (requires alphascout_ml.py)')
    args = ap.parse_args()
    if args.ml:
        main_ml()
    else:
        main()
