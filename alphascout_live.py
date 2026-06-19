#!/usr/bin/env python3
"""
AlphaScout Live  ·  Alpaca Paper/Live Trading  +  ML Signals
═════════════════════════════════════════════════════════════

EXECUTION FLOW  (runs daily at 16:15 ET after market close)
───────────────────────────────────────────────────────────
1. Fetch last HIST_BARS daily bars from Alpaca Data API
2. Re-run full mathematical + ML pipeline (same as backtest)
3. LSTM → regime scalar  (replaces Kalman-velocity threshold)
4. XGBoost → per-stock signal probability (0–1)
5. Score every stock:  Kalman + OU + momentum + Kelly + XGB
6. Rank by risk-adjusted Sharpe momentum
7. Reconcile target portfolio with current Alpaca positions
8. Submit market orders (day orders, fill at next open)
9. Log all activity to alphascout_live.log

FREE-TIER DATA NOTES  (Alpaca Basic Plan)
───────────────────────────────────────────
• Historical bars (IEX feed) — free, no delay, available after ~16:00 ET
• Live quotes — 15-min delayed on free plan
• Strategy uses only end-of-day bars → no dependency on live quotes
• We fetch data at 16:15 ET, ensuring today's close is included
• Set DELAY_GUARD = True to use only bars up to yesterday (extra safety)

INSTALL
───────
  pip install alpaca-py schedule torch xgboost scikit-learn

USAGE
─────
  python alphascout_live.py --train      # train ML models from scratch
  python alphascout_live.py --now        # run one cycle immediately
  python alphascout_live.py --schedule   # start daily 16:15 ET scheduler
  python alphascout_live.py --live       # switch from paper to live account

API KEYS  (set as env vars or edit CONFIG below)
  export ALPACA_API_KEY="your-key"
  export ALPACA_API_SECRET="your-secret"
"""

from __future__ import annotations
import os, sys, time, logging, warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Alpaca ────────────────────────────────────────────────────────────────────
try:
    from alpaca.trading.client   import TradingClient
    from alpaca.trading.requests import (MarketOrderRequest,
                                          GetOrdersRequest,
                                          ClosePositionRequest)
    from alpaca.trading.enums    import OrderSide, TimeInForce, OrderStatus
    from alpaca.data.historical  import StockHistoricalDataClient
    from alpaca.data.requests    import StockBarsRequest
    from alpaca.data.timeframe   import TimeFrame
except ImportError:
    print('Install alpaca-py:  pip install alpaca-py'); sys.exit(1)

try:
    import schedule
except ImportError:
    print('Install schedule:  pip install schedule'); sys.exit(1)

# ── Local modules ─────────────────────────────────────────────────────────────
# alphascout_backtest.py and alphascout_ml.py must be in the same directory
sys.path.insert(0, os.path.dirname(__file__))
from alphascout_backtest import (
    KalmanFilter, OUProcess, HalfKelly, EWMAVol,
    ATR_PERIOD, EWMA_LAMBDA, KELLY_WINDOW, KELLY_FRACTION,
    OU_FIT_WINDOW, MAX_POS_FRAC,
)
from alphascout_ml import (
    MLPipeline, build_ml_features, score_row_ml, regime_from_ml,
    LSTM_FEATURE_COLS,
)

ET = ZoneInfo('America/New_York')

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — edit or set via env vars
# ══════════════════════════════════════════════════════════════════════════════

API_KEY    = os.getenv('ALPACA_API_KEY',    'YOUR_KEY_HERE')
API_SECRET = os.getenv('ALPACA_API_SECRET', 'YOUR_SECRET_HERE')
PAPER      = True                 # always start with paper=True

# Universe — keep identical to backtest to reuse ML models
UNIVERSE  = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AMD',
    'AVGO', 'NFLX', 'JPM',  'V',    'UNH',  'LLY',   'MELI', 'PLTR',
    'SHOP', 'MSTR', 'COIN', 'SOFI',
]
BENCHMARK = 'SPY'

# Risk limits
MAX_POSITIONS   = 5
MAX_POS_FRAC    = 0.20              # more conservative live: 20%
MIN_EQUITY_USD  = 10_000            # hard halt below this
STOP_ATR_MULT   = 2.5
MAX_PORT_DD     = 0.12              # halt trading if portfolio DD > 12%

# ML
MODEL_DIR       = './alphascout_models'    # where models are saved/loaded
ML_ENTRY_SCORE  = 7                        # 0–12 scale (base 0–9 + ML 0–3)
ML_EXIT_SCORE   = 3

# Data
HIST_BARS       = 350               # bars for indicator warm-up (>252 + LSTM seq)
DELAY_GUARD     = False             # True = skip today's bar (extra conservative)
DATA_FEED       = 'iex'             # 'iex' = free plan; 'sip' = paid plan

# Orders
MIN_ORDER_USD   = 200               # don't submit orders below this notional
MAX_ORDER_USD   = 50_000            # single-order cap

# Logging
LOG_FILE        = './alphascout_live.log'
RUN_TIME_ET     = '16:15'           # scheduler run time (24h ET)

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-7s │ %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger('AlphaScout.Live')


# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHER
# ══════════════════════════════════════════════════════════════════════════════

class AlpacaDataFetcher:
    """
    Fetches daily OHLCV bars from Alpaca's Data API.

    Free-tier notes
    ────────────────
    • Uses IEX feed (free) — same data as yfinance for daily bars
    • Historical data has no delay → safe to use today's close
    • DELAY_GUARD=True cuts off today's bar as extra insurance
    • Alpaca rate limit: 200 req/min on free plan → batch requests
    """

    def __init__(self):
        self.client = StockHistoricalDataClient(API_KEY, API_SECRET)

    def fetch(self, tickers: List[str], n_bars: int = HIST_BARS
              ) -> Dict[str, pd.DataFrame]:
        """
        Returns {ticker: DataFrame(open, high, low, close, volume)}
        with at least n_bars rows per ticker where available.
        """
        today    = datetime.now(ET).date()
        # Fetch extra calendar days to account for weekends/holidays
        start_dt = today - timedelta(days=int(n_bars * 1.7))
        end_dt   = (today - timedelta(days=1)) if DELAY_GUARD else today

        log.info(f'Fetching bars: {start_dt} → {end_dt}  '
                 f'(~{n_bars} bars, {len(tickers)} tickers)')

        all_t   = list(set(tickers + [BENCHMARK]))
        # Chunk to stay within rate limits
        chunk   = 50
        batches = [all_t[i: i+chunk] for i in range(0, len(all_t), chunk)]

        result: Dict[str, pd.DataFrame] = {}
        for batch in batches:
            try:
                req  = StockBarsRequest(
                    symbol_or_symbols = batch,
                    timeframe         = TimeFrame.Day,
                    start             = datetime(start_dt.year, start_dt.month, start_dt.day),
                    end               = datetime(end_dt.year,   end_dt.month,   end_dt.day),
                    feed              = DATA_FEED,
                )
                bars = self.client.get_stock_bars(req).df
                if bars.empty:
                    log.warning(f'Empty response for batch {batch[:3]}…')
                    continue

                # Normalise MultiIndex → flat dict
                if isinstance(bars.index, pd.MultiIndex):
                    for t in batch:
                        try:
                            df = bars.xs(t, level='symbol').copy()
                        except KeyError:
                            log.warning(f'  {t}: not in response')
                            continue
                        df = self._normalise(df, t).tail(n_bars)
                        if len(df) >= 50:
                            result[t] = df
                else:
                    for t in batch:
                        sub = bars[bars.get('symbol', pd.Series()) == t].copy()
                        if sub.empty: continue
                        df = self._normalise(sub, t).tail(n_bars)
                        if len(df) >= 50:
                            result[t] = df
            except Exception as e:
                log.error(f'Fetch error batch {batch[:3]}…: {e}')

        log.info(f'Data ready: {len(result)}/{len(all_t)} tickers')
        return result

    @staticmethod
    def _normalise(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        rename = {'open':'open','high':'high','low':'low',
                  'close':'close','volume':'volume',
                  'Open':'open','High':'high','Low':'low',
                  'Close':'close','Volume':'volume'}
        df = df.rename(columns=rename)
        keep = ['open','high','low','close','volume']
        df   = df[[c for c in keep if c in df.columns]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df.dropna()


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (identical pipeline to backtest)
# ══════════════════════════════════════════════════════════════════════════════

def compute_features_live(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Runs the exact same mathematical pipeline as alphascout_backtest.compute_features.
    Keeps the strategy behaviour fully consistent between backtest and live.
    """
    kf    = KalmanFilter()
    ewma  = EWMAVol()
    kelly = HalfKelly()
    out: Dict[str, pd.DataFrame] = {}

    for t, df in data.items():
        d = df.copy()
        c = d['close']; r = c.pct_change()

        # Kalman filter on log-prices
        lp = np.log(c.values.astype(float))
        kf._reset()
        lv, vel, innov = kf.filter_series(lp)
        d['kf_level'] = lv; d['kf_vel'] = vel
        d['kf_price'] = np.exp(lv); d['kf_innov'] = innov

        # OU z-score on Kalman residuals
        residual = pd.Series(lp - lv, index=c.index, name='resid')
        d['ou_z'] = OUProcess.rolling_zscore(residual, window=OU_FIT_WINDOW)

        # ATR
        tr = pd.concat([
            d['high'] - d['low'],
            (d['high'] - c.shift()).abs(),
            (d['low']  - c.shift()).abs(),
        ], axis=1).max(axis=1)
        d['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

        # EWMA-GARCH vol regime
        d = d.join(ewma.compute(r))

        # Half-Kelly fraction
        d['kelly_f'] = kelly.fractions(r)

        # Sharpe momentum
        d['ret_20']   = c.pct_change(20)
        d['vol_20']   = r.rolling(20).std() * np.sqrt(252)
        d['sharpe_m'] = (d['ret_20'] * 252 / 20.0) / (d['vol_20'] + 1e-9)
        d['sharpe_m'] = d['sharpe_m'].clip(-3.0, 10.0)

        # MACD, volume ratio
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        d['macd']      = ema12 - ema26
        d['macd_sig']  = d['macd'].ewm(span=9, adjust=False).mean()
        d['vol_ratio'] = d['volume'] / d['volume'].rolling(20).mean()
        d['ret_1']     = c.pct_change(1)

        out[t] = d

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  ALPACA TRADING CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class AlpacaTrader:
    """Thin wrapper around Alpaca TradingClient with safety checks."""

    def __init__(self):
        self.client = TradingClient(API_KEY, API_SECRET, paper=PAPER)

    def account(self) -> Dict:
        acc = self.client.get_account()
        eq  = float(acc.equity)
        leq = float(acc.last_equity)
        return {
            'equity'        : eq,
            'cash'          : float(acc.cash),
            'buying_power'  : float(acc.buying_power),
            'pnl_today'     : eq - leq,
            'pnl_pct_today' : (eq - leq) / leq if leq > 0 else 0.0,
        }

    def positions(self) -> Dict[str, Dict]:
        result = {}
        for p in self.client.get_all_positions():
            result[p.symbol] = {
                'qty'          : float(p.qty),
                'market_value' : float(p.market_value),
                'avg_cost'     : float(p.avg_entry_price),
                'unrealised_pnl': float(p.unrealized_pl),
                'unrealised_pct': float(p.unrealized_plpc),
            }
        return result

    def cancel_open_orders(self):
        self.client.cancel_orders()
        log.info('All pending orders cancelled.')

    def close_position(self, ticker: str, reason: str = ''):
        try:
            self.client.close_position(ticker)
            log.info(f'CLOSE  {ticker:6s}  [{reason}]')
        except Exception as e:
            log.error(f'Close failed {ticker}: {e}')

    def submit_market_order(self, ticker: str, qty: int, side: str,
                            reason: str = '') -> bool:
        """
        Submit a market order.  side = 'buy' or 'sell'.
        Returns True on success.
        """
        if qty < 1:
            log.warning(f'SKIP {side} {ticker}: qty={qty}')
            return False
        try:
            req = MarketOrderRequest(
                symbol        = ticker,
                qty           = qty,
                side          = OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                time_in_force = TimeInForce.DAY,
            )
            order = self.client.submit_order(req)
            log.info(f'{side.upper():4s}  {qty:5d}  {ticker:6s}  '
                     f'[{reason}]  → {order.id[:8]}')
            return True
        except Exception as e:
            log.error(f'Order error {side} {qty} {ticker}: {e}')
            return False

    def portfolio_drawdown(self, equity: float, high_water: float) -> float:
        """Current DD from high-water mark."""
        if high_water <= 0: return 0.0
        return max(0.0, (high_water - equity) / high_water)


# ══════════════════════════════════════════════════════════════════════════════
#  HIGH-WATER MARK TRACKER  (persistent across sessions)
# ══════════════════════════════════════════════════════════════════════════════

class HWMTracker:
    PATH = './alphascout_hwm.txt'

    def get(self) -> float:
        try:
            with open(self.PATH) as f:
                return float(f.read().strip())
        except Exception:
            return 0.0

    def update(self, equity: float):
        hwm = self.get()
        if equity > hwm:
            with open(self.PATH, 'w') as f:
                f.write(str(equity))


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

class LiveStrategy:
    """
    Daily execution cycle:  fetch → features → ML → score → trade.

    Two modes:
      training  = True  : rebuild ML models from scratch (slow, run weekly/monthly)
      training  = False : load pre-trained models from MODEL_DIR (fast, run daily)
    """

    def __init__(self):
        self.data_fetcher = AlpacaDataFetcher()
        self.trader       = AlpacaTrader()
        self.hwm          = HWMTracker()
        self.pipeline     = MLPipeline(use_akf=False)

    # ── ML model management ───────────────────────────────────────────────

    def ensure_models(self, data: Dict[str, pd.DataFrame],
                      features: Dict[str, pd.DataFrame], force_retrain: bool = False):
        """Load models from disk if they exist, else train from scratch."""
        lstm_path = os.path.join(MODEL_DIR, 'lstm_regime.pt')
        has_models = os.path.exists(lstm_path)

        if force_retrain or not has_models:
            log.info('Training ML models from scratch …')
            ml_data = build_ml_features(features)
            # Inject ML features into features dict
            for t in ml_data:
                features[t] = ml_data[t]
            self.pipeline.fit(features)
            self.pipeline.save(MODEL_DIR)
        else:
            log.info(f'Loading ML models from {MODEL_DIR}/ …')
            self.pipeline.load(MODEL_DIR, tickers=UNIVERSE, benchmark=BENCHMARK)
            # Still run the augmentation for xgb_prob / regime_prob
            ml_feat = build_ml_features(features)
            spy_df  = ml_feat.get(BENCHMARK)
            for t, df in ml_feat.items():
                # XGB inference on latest bar
                xgb_m = self.pipeline.xgb_models.get(t)
                if xgb_m:
                    prob = xgb_m.predict_latest(df)
                    df['xgb_prob'] = prob if prob is not None else 0.5
                else:
                    df['xgb_prob'] = 0.5
                # LSTM regime broadcast
                if spy_df is not None:
                    df['regime_prob'] = self.pipeline.lstm.predict_prob(spy_df)
                features[t] = df

    # ── signal / portfolio ────────────────────────────────────────────────

    def compute_target(self, features: Dict[str, pd.DataFrame],
                       total_equity: float, regime_scalar: float
                       ) -> Dict[str, float]:
        """
        Returns {ticker: target_dollar_allocation}.
        Scores all stocks and takes the top MAX_POSITIONS by Sharpe momentum.
        """
        scores: Dict[str, int]        = {}
        rows:   Dict[str, pd.Series]  = {}

        for t in UNIVERSE:
            df = features.get(t)
            if df is None or df.empty:
                continue
            row = df.iloc[-1]
            if row.isnull().any():
                continue
            sc = score_row_ml(row, self.pipeline)
            if sc >= ML_ENTRY_SCORE:
                scores[t] = sc
                rows[t]   = row

        # Rank by Sharpe momentum
        ranked = sorted(scores, key=lambda t: -float(rows[t].get('sharpe_m', 0.0)))
        target = ranked[:MAX_POSITIONS]

        result = {}
        for t in target:
            row        = rows[t]
            kelly_f    = float(row.get('kelly_f',    MAX_POS_FRAC / MAX_POSITIONS))
            vol_scl    = float(row.get('vol_scalar', 1.0))
            frac       = min(kelly_f * vol_scl * regime_scalar, MAX_POS_FRAC)
            frac       = max(frac, 0.05)
            result[t]  = min(total_equity * frac, MAX_ORDER_USD)

        return result

    # ── execution ─────────────────────────────────────────────────────────

    def execute_rebalance(self, target: Dict[str, float],
                          current: Dict[str, Dict],
                          features: Dict[str, pd.DataFrame],
                          total_equity: float):
        """
        Reconcile target vs current positions, submit minimum-churn orders.
        """
        # 1. Exit stale positions (not in target, score too low)
        for t in list(current.keys()):
            df = features.get(t)
            sc = 0
            if df is not None and not df.empty:
                sc = score_row_ml(df.iloc[-1], self.pipeline)
            if t not in target and sc <= ML_EXIT_SCORE:
                self.trader.close_position(t, f'exit(score={sc})')

        # 2. Stop-loss check (soft — Alpaca bracket orders are better for this)
        for t, pos in current.items():
            df = features.get(t)
            if df is None or df.empty:
                continue
            row   = df.iloc[-1]
            stop  = pos['avg_cost'] - STOP_ATR_MULT * float(row.get('atr', 0.0))
            price = float(row.get('close', 0.0))
            if price > 0 and price < stop:
                log.warning(f'STOP  {t}: close={price:.2f} < stop={stop:.2f}')
                self.trader.close_position(t, 'stop_loss')

        # 3. Enter / size-up new positions
        current_fresh = self.trader.positions()  # re-fetch after exits
        bp = self.trader.account()['buying_power']

        for t, dollar_val in target.items():
            df = features.get(t)
            if df is None or df.empty:
                continue
            price = float(df.iloc[-1].get('close', 0.0))
            if price <= 0:
                continue

            if t in current_fresh:
                # Already have position — check if we should adjust size
                curr_val   = current_fresh[t]['market_value']
                delta_val  = dollar_val - curr_val
                if abs(delta_val) < MIN_ORDER_USD:
                    continue           # within tolerance, skip
                side = 'buy' if delta_val > 0 else 'sell'
                qty  = max(1, int(abs(delta_val) / price))
                if side == 'buy' and qty * price > bp:
                    qty = int(bp / price)
                if qty >= 1:
                    self.trader.submit_market_order(t, qty, side,
                                                    reason='rebalance')
            else:
                # New position
                if dollar_val < MIN_ORDER_USD:
                    continue
                qty = int(min(dollar_val, bp * 0.95) / price)
                if qty >= 1:
                    self.trader.submit_market_order(t, qty, 'buy',
                                                    reason='new_entry')

    # ── main cycle ────────────────────────────────────────────────────────

    def run_cycle(self, force_retrain: bool = False):
        """Execute one complete daily cycle."""
        now_et = datetime.now(ET)
        log.info('━' * 58)
        log.info(f' AlphaScout Live  —  {now_et.strftime("%Y-%m-%d %H:%M ET")}')
        log.info(f' Mode: {"PAPER" if PAPER else "⚠  LIVE"}')
        log.info('━' * 58)

        # ── 1. Account state ─────────────────────────────────────────────
        acc = self.trader.account()
        log.info(f'Equity: ${acc["equity"]:>10,.0f}   '
                 f'PnL today: ${acc["pnl_today"]:>+8,.0f}  '
                 f'({acc["pnl_pct_today"]*100:+.2f}%)')

        if acc['equity'] < MIN_EQUITY_USD:
            log.critical(f'Equity below floor (${MIN_EQUITY_USD:,.0f}). HALTED.')
            return

        # ── 2. Portfolio drawdown check ──────────────────────────────────
        self.hwm.update(acc['equity'])
        dd = self.trader.portfolio_drawdown(acc['equity'], self.hwm.get())
        if dd > MAX_PORT_DD:
            log.critical(f'Portfolio DD={dd*100:.1f}% > {MAX_PORT_DD*100:.0f}%. '
                         f'Liquidating and halting.')
            for t in self.trader.positions():
                self.trader.close_position(t, 'max_drawdown_halt')
            return

        # ── 3. Fetch data ─────────────────────────────────────────────────
        raw      = self.data_fetcher.fetch(UNIVERSE)
        if len(raw) < 5:
            log.error('Insufficient tickers — aborting'); return

        # ── 4. Feature engineering ────────────────────────────────────────
        features = compute_features_live(raw)

        # ── 5. ML models (load or train) ──────────────────────────────────
        self.ensure_models(raw, features, force_retrain=force_retrain)

        # ── 6. Market regime via LSTM ─────────────────────────────────────
        spy_df      = features.get(BENCHMARK)
        regime_prob = self.pipeline.lstm.predict_prob(spy_df) if spy_df is not None else 0.6
        regime_scl  = self.pipeline.lstm.regime_scalar(regime_prob)
        log.info(f'Regime:  P(bull)={regime_prob:.2f}  →  '
                 f'exposure={regime_scl:.2f}  '
                 f'({"BULL" if regime_prob>=0.7 else "BEAR" if regime_prob<0.4 else "NEUTRAL"})')

        # ── 7. Risk-off: liquidate all if regime says cash ────────────────
        if regime_scl == 0.0:
            log.info('REGIME OFF — liquidating all positions')
            for t in self.trader.positions():
                self.trader.close_position(t, 'regime_off')
            return

        # ── 8. Compute target portfolio ───────────────────────────────────
        total_eq = acc['equity']
        target   = self.compute_target(features, total_eq, regime_scl)
        log.info(f'Target portfolio: {list(target.keys())} '
                 f'(regime_scalar={regime_scl:.2f})')

        # ── 9. Log per-stock signals ──────────────────────────────────────
        for t, dol in target.items():
            df = features.get(t)
            if df is None: continue
            row = df.iloc[-1]
            log.info(f'  {t:6s}  score={score_row_ml(row, self.pipeline):2d}  '
                     f'sharpe_m={row.get("sharpe_m",0):.2f}  '
                     f'ou_z={row.get("ou_z",0):.2f}  '
                     f'xgb={row.get("xgb_prob",0.5):.2f}  '
                     f'alloc=${dol:,.0f}')

        # ── 10. Execute ────────────────────────────────────────────────────
        current = self.trader.positions()
        self.execute_rebalance(target, current, features, total_eq)
        log.info('Cycle complete.\n')

    # ── scheduler ─────────────────────────────────────────────────────────

    def start(self, run_time: str = RUN_TIME_ET, force_retrain: bool = False):
        """
        Start the daily scheduler.
        Runs run_cycle() at run_time ET on market days.
        """
        log.info(f'Scheduler started — running daily at {run_time} ET')
        log.info(f'GPU: {_device_str()}')

        def _job():
            now = datetime.now(ET)
            # Skip if not a weekday (rough check — Alpaca won't process anyway)
            if now.weekday() >= 5:
                log.info('Weekend — skipping.')
                return
            self.run_cycle(force_retrain=force_retrain)
            force_retrain = False   # only force-retrain on first run

        schedule.every().monday.at(run_time).do(_job)
        schedule.every().tuesday.at(run_time).do(_job)
        schedule.every().wednesday.at(run_time).do(_job)
        schedule.every().thursday.at(run_time).do(_job)
        schedule.every().friday.at(run_time).do(_job)

        while True:
            schedule.run_pending()
            time.sleep(30)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _device_str() -> str:
    import torch
    if torch.cuda.is_available():
        return (f'CUDA — {torch.cuda.get_device_name(0)} '
                f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
    return 'CPU (no CUDA)'


def print_banner():
    log.info('═' * 58)
    log.info(' AlphaScout Live  ·  Kalman–Kelly–OU + ML')
    log.info(f' Compute: {_device_str()}')
    log.info(f' Mode:    {"PAPER TRADING" if PAPER else "⚠  LIVE TRADING"}')
    log.info(f' Models:  {MODEL_DIR}/')
    log.info('═' * 58)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description='AlphaScout Live Trading')
    p.add_argument('--train',    action='store_true',
                   help='Force retrain all ML models from scratch')
    p.add_argument('--now',      action='store_true',
                   help='Run one cycle immediately then exit')
    p.add_argument('--schedule', action='store_true',
                   help=f'Start daily scheduler at {RUN_TIME_ET} ET')
    p.add_argument('--live',     action='store_true',
                   help='Use LIVE account (default: paper)')
    p.add_argument('--status',   action='store_true',
                   help='Print account status and current positions then exit')
    args = p.parse_args()

    if args.live:
        global PAPER
        PAPER = False
        log.warning('⚠  LIVE TRADING MODE — real money at risk')

    print_banner()

    if not any([args.train, args.now, args.schedule, args.status]):
        p.print_help()
        return

    strategy = LiveStrategy()

    if args.status:
        acc  = strategy.trader.account()
        pos  = strategy.trader.positions()
        hwm  = strategy.hwm.get()
        dd   = strategy.trader.portfolio_drawdown(acc['equity'], hwm)
        log.info(f'Equity:       ${acc["equity"]:>12,.2f}')
        log.info(f'Cash:         ${acc["cash"]:>12,.2f}')
        log.info(f'Buying power: ${acc["buying_power"]:>12,.2f}')
        log.info(f'PnL today:    ${acc["pnl_today"]:>+12,.2f}  '
                 f'({acc["pnl_pct_today"]*100:+.2f}%)')
        log.info(f'High-water:   ${hwm:>12,.2f}')
        log.info(f'Drawdown:     {dd*100:.1f}%')
        if pos:
            log.info(f'Positions ({len(pos)}):')
            for t, v in pos.items():
                log.info(f'  {t:6s}  qty={v["qty"]:.0f}  '
                         f'value=${v["market_value"]:,.0f}  '
                         f'pnl=${v["unrealised_pnl"]:+,.0f}  '
                         f'({v["unrealised_pct"]*100:+.1f}%)')
        else:
            log.info('No open positions.')
        return

    if args.now:
        strategy.run_cycle(force_retrain=args.train)
    elif args.schedule:
        strategy.start(force_retrain=args.train)


if __name__ == '__main__':
    main()
