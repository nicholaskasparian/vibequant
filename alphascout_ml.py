#!/usr/bin/env python3
"""
AlphaScout ML  ·  GPU-Accelerated Machine Learning Signals
══════════════════════════════════════════════════════════

THREE ML COMPONENTS — each mathematically distinct, GPU-accelerated
────────────────────────────────────────────────────────────────────

1. WALK-FORWARD XGBOOST  (classification signal)
   ─────────────────────────────────────────────
   Additive ensemble:  F(x) = Σ_{k=1}^{K} f_k(x),   f_k ∈ CART

   At boosting step k, minimise the 2nd-order Taylor expansion:
     Ẽ_k = Σ_i [ g_i f_k(x_i) + ½ h_i f_k(x_i)² ] + Ω(f_k)

   Gradients of log-loss  ℓ(y, p) = −y log p − (1−y) log(1−p):
     g_i  = p̂_i − y_i              (first-order gradient)
     h_i  = p̂_i (1 − p̂_i)         (second-order / Hessian)

   Optimal leaf weight at node j:   w_j* = −G_j / (H_j + λ)
   Split gain for candidate split:  ΔL = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ)
                                              − (G_L+G_R)²/(H_L+H_R+λ) ] − γ

   GPU histogram algorithm (XGBoost ≥ 2.0, device='cuda'):
   Reduces feature-split scanning from O(n·d) → O(b·d / g)
   where b=histogram bins, d=features, g=GPU parallelism.

   Applied as walk-forward binary classifier:
   Target y_t = 𝟙[ close_{t+5} > close_t ]  (5-day forward return direction)
   Training: expanding window, retrain every 21 bars.
   No lookahead: features at t use only data ≤ t; label requires data at t+5.

2. LSTM REGIME CLASSIFIER  (market regime signal)
   ─────────────────────────────────────────────
   At each step t, LSTM transitions (σ = sigmoid, ⊙ = Hadamard):

     f_t = σ( W_f [h_{t-1}, x_t] + b_f )      forget gate
     i_t = σ( W_i [h_{t-1}, x_t] + b_i )      input gate
     c̃_t = tanh( W_c [h_{t-1}, x_t] + b_c )  candidate cell
     c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t        cell state
     o_t = σ( W_o [h_{t-1}, x_t] + b_o )      output gate
     h_t = o_t ⊙ tanh(c_t)                     hidden state

   Key property: gradient ∂L/∂c_{t-k} = ∂L/∂c_t · Π_{j=t-k+1}^{t} f_j
   Since f_j ∈ (0,1) but doesn't accumulate multiplicatively (additive cell
   update), LSTMs avoid the vanishing-gradient problem of vanilla RNNs.

   Input: 60-day window of SPY [daily return, EWMA vol, KF velocity,
                                 OU z-score, Sharpe momentum, vol scalar]
   Target: 𝟙[ forward 20d SPY return > 0 ]   (bull-regime probability)
   Output: P(bull) ∈ [0, 1]

3. ADAPTIVE KALMAN FILTER  (deep state space model)
   ──────────────────────────────────────────────────
   Standard KF fixes Q, R.  Here we learn them as functions of context:

     Q_t = f_Q( context_t ; θ_Q )     — process noise (via Softplus MLP)
     R_t = f_R( context_t ; θ_R )     — observation noise

   context_t = [ σ_ewm,t / σ_lt,t ,  |r_{t-5:t}| ,  |ν_{t-5:t}| ,  σ_ewm,t ]

   The Kalman recursion is implemented as differentiable torch operations:
     Predict:   x̂⁻_t = F x̂_{t-1},      P⁻_t = F P_{t-1} Fᵀ + Q_t
     Innovate:  ν_t = y_t − H x̂⁻_t,    S_t = H P⁻_t Hᵀ + R_t
     Update:    K_t = P⁻_t Hᵀ / S_t
                x̂_t = x̂⁻_t + K_t ν_t
                P_t = (I − K_t H) P⁻_t

   Training loss  (negative log marginal likelihood):
     L(θ) = ½ Σ_t [ ν_t² / S_t + log S_t ]

   This is the EXACT MLE for linear Gaussian SSMs.
   Gradient ∂L/∂θ backpropagates through all Kalman steps on GPU.
   Result: Q, R that adapt to market volatility regimes automatically.

DATA SPLIT RULES (no lookahead bias)
────────────────────────────────────
  Feature at bar t  → uses only price/indicator data ≤ t
  XGB target at t   → uses close_{t+5}  (excluded from training until t+5 is observed)
  LSTM target at t  → uses close_{t+20} (same rule)
  Adaptive KF       → pre-trained on first PRETRAIN_BARS bars; rolled forward
"""

from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import logging, os, pickle

log = logging.getLogger('AlphaScout.ML')

# ── GPU detection ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    log.info(f'GPU: {torch.cuda.get_device_name(0)}  '
             f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)')
else:
    log.info('CUDA not found — running on CPU (GPU will be used on your machine)')

import xgboost as xgb
XGB_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# XGBoost
XGB_N_ESTIMATORS  = 500
XGB_MAX_DEPTH      = 4          # shallow: reduces overfitting
XGB_LR             = 0.03
XGB_SUBSAMPLE      = 0.8
XGB_COL_SAMPLE     = 0.8
XGB_L2             = 1.0        # L2 regularisation on leaf weights
XGB_EARLY_STOP     = 30
XGB_LOOKAHEAD      = 5          # bars forward for binary target
XGB_MIN_TRAIN      = 252        # minimum bars before first model
XGB_RETRAIN_EVERY  = 21         # retrain monthly

# LSTM
LSTM_SEQ_LEN       = 60         # 60-bar lookback window
LSTM_HIDDEN        = 128
LSTM_LAYERS        = 2
LSTM_DROPOUT       = 0.25
LSTM_LOOKAHEAD     = 20         # 20-day forward SPY return for label
LSTM_EPOCHS        = 80
LSTM_BATCH         = 64
LSTM_LR            = 3e-4
LSTM_MIN_TRAIN     = 400        # bars before first training

# Adaptive Kalman
AKF_PRETRAIN_BARS  = 300
AKF_EPOCHS         = 200
AKF_LR             = 1e-3
AKF_CONTEXT_DIM    = 4          # context features
AKF_HIDDEN         = 16

# Features used by XGBoost
XGB_FEATURE_COLS = [
    'kf_vel',         # Kalman velocity (log-return scale)
    'price_vs_kf',    # close / kf_price − 1  (deviation from KF)
    'kf_innov_norm',  # normalised Kalman innovation
    'ou_z',           # OU z-score
    'kelly_f',        # Kelly fraction
    'sharpe_m',       # 20-day Sharpe momentum
    'vol_scalar',     # EWMA-GARCH regime scalar
    'vol_ratio',      # volume / 20-day avg
    'macd_xover',     # MACD − Signal (signed)
    'ret_5',          # 5-day return
    'ret_20',         # 20-day return
    'vol_20',         # 20-day realised vol (annualised)
    'atr_norm',       # ATR / close
]

# SPY features used by LSTM regime classifier
LSTM_FEATURE_COLS = [
    'ret_1',          # daily return
    'vol_ratio',      # volume ratio
    'kf_vel',         # Kalman velocity
    'ou_z',           # OU z-score
    'sharpe_m',       # Sharpe momentum
    'vol_scalar',     # vol regime scalar
]

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (builds ML-ready columns from base features)
# ══════════════════════════════════════════════════════════════════════════════

def build_ml_features(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Extends the base feature DataFrames (from alphascout_backtest.compute_features)
    with additional derived columns needed by the ML models.
    All features are strictly backward-looking (no lookahead).
    """
    out = {}
    for t, df in data.items():
        d  = df.copy()
        c  = d['close']

        # Derived features for XGBoost ─────────────────────────────────────
        # Price deviation from Kalman smoothed level (normalised)
        d['price_vs_kf'] = c / d['kf_price'].replace(0, np.nan) - 1.0

        # Normalised Kalman innovation (innovation / ATR proxy)
        atr_proxy = d['atr'].replace(0, np.nan)
        d['kf_innov_norm'] = d['kf_innov'] / atr_proxy.where(atr_proxy > 0, np.nan)

        # 1-day return (used by LSTM)
        d['ret_1']   = c.pct_change(1)
        d['ret_5']   = c.pct_change(5)

        # ATR normalised
        d['atr_norm'] = d['atr'] / c.replace(0, np.nan)

        # MACD crossover: signed difference
        d['macd_xover'] = d['macd'] - d['macd_sig']

        # Clip to remove extreme outliers that can destabilise tree splits
        clip_cols = ['kf_vel', 'ou_z', 'sharpe_m', 'kf_innov_norm', 'price_vs_kf']
        for col in clip_cols:
            if col in d.columns:
                lo, hi = d[col].quantile(0.01), d[col].quantile(0.99)
                d[col] = d[col].clip(lo, hi)

        out[t] = d
    return out


def _target_series(close: pd.Series, lookahead: int) -> pd.Series:
    """
    Binary target: y_t = 1 if close_{t+lookahead} > close_t, else 0.
    Last `lookahead` rows are NaN (future not yet observed).
    """
    fwd = close.shift(-lookahead) / close - 1.0
    return (fwd > 0.0).astype(float)


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 1: WALK-FORWARD XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

class WalkForwardXGB:
    """
    Per-ticker walk-forward XGBoost classifier.

    Expanding training window: never uses future data.
    GPU-accelerated via XGBoost's CUDA histogram algorithm.
    Returns P(5-day forward return > 0) at each bar.
    """

    def __init__(self,
                 lookahead:      int   = XGB_LOOKAHEAD,
                 min_train:      int   = XGB_MIN_TRAIN,
                 retrain_every:  int   = XGB_RETRAIN_EVERY,
                 feature_cols:   List  = XGB_FEATURE_COLS):
        self.lookahead     = lookahead
        self.min_train     = min_train
        self.retrain_every = retrain_every
        self.feature_cols  = feature_cols
        self._model: Optional[xgb.XGBClassifier] = None
        self.feature_importance_: Optional[pd.Series] = None

    def _make_model(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators        = XGB_N_ESTIMATORS,
            max_depth           = XGB_MAX_DEPTH,
            learning_rate       = XGB_LR,
            subsample           = XGB_SUBSAMPLE,
            colsample_bytree    = XGB_COL_SAMPLE,
            reg_lambda          = XGB_L2,
            device              = XGB_DEVICE,      # 'cuda' or 'cpu'
            tree_method         = 'hist',
            eval_metric         = 'logloss',
            early_stopping_rounds = XGB_EARLY_STOP,
            verbosity           = 0,
            random_state        = 42,
        )

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Walk-forward fit + predict on full DataFrame.
        df must contain all XGB_FEATURE_COLS and 'close'.
        Returns Series of P(positive_5d_return), aligned with df.index.
        """
        feat_cols = [c for c in self.feature_cols if c in df.columns]
        target    = _target_series(df['close'], self.lookahead)
        n         = len(df)
        probs     = pd.Series(np.nan, index=df.index, name='xgb_prob')

        model = None
        for i in range(self.min_train, n - self.lookahead):
            retrain = (model is None) or \
                      ((i - self.min_train) % self.retrain_every == 0)
            if retrain:
                # Training set: bars 0 .. i−lookahead (labels known)
                end = i - self.lookahead
                X_tr = df[feat_cols].iloc[:end].values.astype(np.float32)
                y_tr = target.iloc[:end].values.astype(np.float32)

                # Drop rows with any NaN
                valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
                X_tr, y_tr = X_tr[valid], y_tr[valid]
                if len(X_tr) < 100:
                    continue

                # 10% of training set as eval for early stopping
                split   = max(1, int(len(X_tr) * 0.1))
                X_val   = X_tr[-split:];  y_val = y_tr[-split:]
                X_tr    = X_tr[:-split];  y_tr  = y_tr[:-split]

                model = self._make_model()
                model.fit(X_tr, y_tr,
                          eval_set=[(X_val, y_val)],
                          verbose=False)

                if self.feature_importance_ is None:
                    self.feature_importance_ = pd.Series(
                        model.feature_importances_, index=feat_cols,
                        name='importance'
                    ).sort_values(ascending=False)
                    log.info(f'  XGB feature importance (top 5):\n'
                             f'{self.feature_importance_.head(5)}')
                self._model = model

            # Predict at bar i
            x = df[feat_cols].iloc[i:i+1].values.astype(np.float32)
            if not np.isnan(x).any():
                probs.iloc[i] = model.predict_proba(x)[0, 1]

        log.info(f'  Walk-forward XGB: {(~probs.isna()).sum()} predictions')
        return probs

    def predict_latest(self, df: pd.DataFrame) -> Optional[float]:
        """Inference-only: predict on the most recent bar using a fitted model."""
        if self._model is None:
            return None
        feat_cols = [c for c in self.feature_cols if c in df.columns]
        x = df[feat_cols].iloc[-1:].values.astype(np.float32)
        if np.isnan(x).any():
            return None
        return float(self._model.predict_proba(x)[0, 1])

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'model': self._model,
                         'fi': self.feature_importance_}, f)
        log.info(f'XGB saved → {path}')

    def load(self, path: str):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self._model = d['model']
        self.feature_importance_ = d.get('fi')
        log.info(f'XGB loaded ← {path}')


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 2: LSTM REGIME CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class _RegimeNet(nn.Module):
    """
    Stacked LSTM → fully-connected head → sigmoid.
    Input:  (batch, seq_len, n_features)
    Output: (batch, 1)  —  P(bull regime)
    """
    def __init__(self, n_feat: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_feat,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # use final hidden state


class RegimeLSTM:
    """
    LSTM-based market regime classifier trained on SPY.

    Input:  60-bar sequence of 6 SPY features (normalised)
    Output: P(bull) ∈ [0, 1]

    Label:  y_t = 1 if SPY closes higher 20 bars forward  (forward return > 0)
    This avoids any subjective regime definition — pure data-driven.

    Training: rolling-window with 80/20 chronological train/val split.
    GPU:       model.to(DEVICE)  (auto-detects CUDA)
    """

    def __init__(self,
                 seq_len:   int = LSTM_SEQ_LEN,
                 lookahead: int = LSTM_LOOKAHEAD,
                 min_train: int = LSTM_MIN_TRAIN,
                 n_feat:    int = len(LSTM_FEATURE_COLS)):
        self.seq_len   = seq_len
        self.lookahead = lookahead
        self.min_train = min_train
        self.scaler    = StandardScaler()
        self.net: Optional[_RegimeNet] = None
        self._n_feat   = n_feat
        self._trained  = False

    # ── data helpers ──────────────────────────────────────────────────────

    def _make_sequences(self, X: np.ndarray, y: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Sliding-window sequences from flat arrays."""
        xs, ys = [], []
        for i in range(self.seq_len, len(y)):
            xs.append(X[i - self.seq_len: i])
            ys.append(y[i])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    # ── training ──────────────────────────────────────────────────────────

    def train(self, spy_df: pd.DataFrame):
        """
        Train on SPY feature history.
        spy_df must contain LSTM_FEATURE_COLS and 'close'.
        """
        feat_cols = [c for c in LSTM_FEATURE_COLS if c in spy_df.columns]
        if len(feat_cols) < 4 or len(spy_df) < self.min_train + self.lookahead:
            log.warning('LSTM: insufficient data — skipping training')
            return

        X_raw = spy_df[feat_cols].values
        y_raw = _target_series(spy_df['close'], self.lookahead).values

        # Trim to valid rows (both X and y not NaN)
        valid = ~(np.isnan(X_raw).any(axis=1) | np.isnan(y_raw))
        X_raw, y_raw = X_raw[valid], y_raw[valid]

        # Normalise features (fit only on first 80% of data)
        n_tr   = int(len(X_raw) * 0.8)
        self.scaler.fit(X_raw[:n_tr])
        X_norm = self.scaler.transform(X_raw)

        X_seq, y_seq = self._make_sequences(X_norm, y_raw)
        if len(X_seq) < 50:
            log.warning('LSTM: too few sequences'); return

        split    = int(len(X_seq) * 0.8)
        X_tr, y_tr = X_seq[:split],  y_seq[:split]
        X_vl, y_vl = X_seq[split:],  y_seq[split:]

        t_tr = torch.tensor(X_tr).to(DEVICE)
        l_tr = torch.tensor(y_tr).to(DEVICE)
        t_vl = torch.tensor(X_vl).to(DEVICE)
        l_vl = torch.tensor(y_vl).to(DEVICE)

        ds   = TensorDataset(t_tr, l_tr)
        dl   = DataLoader(ds, batch_size=LSTM_BATCH, shuffle=True)

        self.net = _RegimeNet(
            n_feat  = len(feat_cols),
            hidden  = LSTM_HIDDEN,
            n_layers= LSTM_LAYERS,
            dropout = LSTM_DROPOUT,
        ).to(DEVICE)

        opt      = optim.AdamW(self.net.parameters(), lr=LSTM_LR, weight_decay=1e-4)
        sched    = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=LSTM_EPOCHS)
        loss_fn  = nn.BCELoss()
        best_val = np.inf
        patience = 15
        patience_counter = 0

        for epoch in range(LSTM_EPOCHS):
            self.net.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            sched.step()

            # Validation
            self.net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net(t_vl), l_vl).item()
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                patience_counter = 0
                self._best_state = {k: v.clone() for k, v in
                                    self.net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log.info(f'  LSTM early stop at epoch {epoch+1}')
                    break

        self.net.load_state_dict(self._best_state)
        self._n_feat = len(feat_cols)
        self._trained = True
        log.info(f'  LSTM trained — val_loss={best_val:.4f}  '
                 f'device={DEVICE.type}')

    def predict_prob(self, spy_df: pd.DataFrame) -> float:
        """
        Returns P(bull) for the most recent available date.
        Uses the last seq_len bars from spy_df.
        """
        if not self._trained or self.net is None:
            return 0.7   # optimistic default if not trained

        feat_cols = [c for c in LSTM_FEATURE_COLS if c in spy_df.columns]
        df_tail   = spy_df.tail(self.seq_len * 2)
        X_raw     = df_tail[feat_cols].dropna().values

        if len(X_raw) < self.seq_len:
            return 0.7

        X_norm = self.scaler.transform(X_raw[-self.seq_len:])
        t      = torch.tensor(X_norm[np.newaxis], dtype=torch.float32).to(DEVICE)

        self.net.eval()
        with torch.no_grad():
            prob = float(self.net(t).item())
        return prob

    def predict_series(self, spy_df: pd.DataFrame) -> pd.Series:
        """Rolling predictions over full history (for backtest integration)."""
        if not self._trained or self.net is None:
            return pd.Series(0.7, index=spy_df.index, name='regime_prob')

        feat_cols = [c for c in LSTM_FEATURE_COLS if c in spy_df.columns]
        X_raw     = spy_df[feat_cols].values
        valid     = ~np.isnan(X_raw).any(axis=1)
        X_norm_all = np.full_like(X_raw, np.nan)
        X_norm_all[valid] = self.scaler.transform(X_raw[valid])

        probs = pd.Series(np.nan, index=spy_df.index, name='regime_prob')
        self.net.eval()

        # Batch for efficiency
        xs = []
        idx = []
        for i in range(self.seq_len, len(spy_df)):
            window = X_norm_all[i - self.seq_len: i]
            if not np.isnan(window).any():
                xs.append(window)
                idx.append(spy_df.index[i])

        if not xs:
            return probs

        batch  = np.array(xs, dtype=np.float32)
        chunks = [batch[j: j+512] for j in range(0, len(batch), 512)]

        all_probs = []
        with torch.no_grad():
            for chunk in chunks:
                t = torch.tensor(chunk).to(DEVICE)
                all_probs.extend(self.net(t).cpu().numpy().tolist())

        for i_val, i_idx in zip(all_probs, idx):
            probs.loc[i_idx] = i_val

        return probs

    def regime_scalar(self, prob: float) -> float:
        """
        Convert P(bull) → position exposure scalar.
        Thresholds chosen to be less whipsaw-prone than binary.
        """
        if prob >= 0.70:   return 1.00   # strong bull
        if prob >= 0.55:   return 0.75   # mild bull
        if prob >= 0.40:   return 0.50   # uncertain
        if prob >= 0.25:   return 0.25   # mild bear
        return 0.00                       # strong bear — full cash

    def save(self, path: str):
        torch.save({
            'net_state' : self.net.state_dict() if self.net else None,
            'scaler'    : self.scaler,
            'n_feat'    : self._n_feat,
            'trained'   : self._trained,
        }, path)
        log.info(f'LSTM saved → {path}')

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.scaler    = ckpt['scaler']
        self._n_feat   = ckpt['n_feat']
        self._trained  = ckpt['trained']
        if ckpt['net_state'] and self._trained:
            self.net = _RegimeNet(
                n_feat  = self._n_feat,
                hidden  = LSTM_HIDDEN,
                n_layers= LSTM_LAYERS,
                dropout = LSTM_DROPOUT,
            ).to(DEVICE)
            self.net.load_state_dict(ckpt['net_state'])
        log.info(f'LSTM loaded ← {path}')


# ══════════════════════════════════════════════════════════════════════════════
#  COMPONENT 3: ADAPTIVE KALMAN FILTER  (Deep State Space Model)
# ══════════════════════════════════════════════════════════════════════════════

class _NoiseNet(nn.Module):
    """
    Small MLP that maps context features → positive noise parameters.
    Uses Softplus activation to ensure positivity: log(1 + exp(x)) > 0.

    Outputs: (q_level, q_velocity, r_obs) — all strictly positive.
    """
    def __init__(self, context_dim: int = AKF_CONTEXT_DIM,
                 hidden: int = AKF_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
            nn.Softplus(),           # ensures all outputs > 0
        )
        # Initialise near the hand-tuned defaults (from backtest)
        nn.init.constant_(self.net[-2].bias, -2.0)   # Softplus(-2) ≈ 0.13

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out = self.net(ctx)          # (batch, 3)
        # Scale to physically meaningful ranges
        q_L = out[:, 0] * 0.01 + 1e-5     # level noise: ~O(1e-4)
        q_V = out[:, 1] * 0.05 + 1e-4     # velocity noise: ~O(1e-3)
        r   = out[:, 2] * 2.0  + 0.1      # observation noise: ~O(1)
        return q_L, q_V, r


class AdaptiveKalman:
    """
    Kalman filter with learned (time-varying) Q and R.

    Training: minimise negative log marginal likelihood on historical log-prices.
    Loss at each step:  ℓ_t = ½ [ ν_t² / S_t + log(S_t) ]
    Total: L(θ) = Σ_t ℓ_t     (backprop through all Kalman steps)

    At inference: replaces the fixed-noise KalmanFilter in alphascout_backtest.py.
    Provides adaptive kf_level and kf_vel with regime-aware noise.
    """

    def __init__(self):
        self.noise_net  = _NoiseNet(AKF_CONTEXT_DIM, AKF_HIDDEN).to(DEVICE)
        self.opt        = optim.Adam(self.noise_net.parameters(), lr=AKF_LR)
        self._trained   = False

    # ── differentiable Kalman step ────────────────────────────────────────

    @staticmethod
    def _kf_step(x: torch.Tensor, P: torch.Tensor,
                 y: torch.Tensor,
                 q_L: torch.Tensor, q_V: torch.Tensor,
                 r: torch.Tensor, dt: float = 1.0
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One Kalman step (all batched torch operations — fully differentiable).
        x: (B,2)  P: (B,2,2)  y: (B,)  returns: x, P, nu, S
        """
        B = x.shape[0]
        F = torch.tensor([[1.0, dt], [0.0, 1.0]],
                         dtype=torch.float32, device=DEVICE)   # (2,2)
        H = torch.tensor([[1.0, 0.0]],
                         dtype=torch.float32, device=DEVICE)   # (1,2)

        # Build Q per-sample
        Q = torch.zeros(B, 2, 2, device=DEVICE)
        Q[:, 0, 0] = q_L
        Q[:, 1, 1] = q_V

        # Predict
        x_p = (F @ x.unsqueeze(-1)).squeeze(-1)          # (B,2)
        P_p = F @ P @ F.T + Q                            # (B,2,2)

        # Innovation
        Hx_p = (H @ x_p.unsqueeze(-1)).squeeze(-1)      # (B,1)
        nu   = y - Hx_p[:, 0]                            # (B,)
        S    = (H @ P_p @ H.T)[:, 0, 0] + r             # (B,)
        K    = (P_p @ H.T)[:, :, 0] / S.unsqueeze(1)    # (B,2)

        # Update
        x_u = x_p + K * nu.unsqueeze(1)
        I   = torch.eye(2, device=DEVICE).unsqueeze(0).expand(B, -1, -1)
        P_u = (I - K.unsqueeze(2) @ H.unsqueeze(0)) @ P_p

        return x_u, P_u, nu, S

    # ── training ──────────────────────────────────────────────────────────

    def train_on(self, df: pd.DataFrame, n_bars: int = AKF_PRETRAIN_BARS):
        """
        Train noise_net to minimise NLML on the first n_bars of df.
        df must contain 'close', 'vol_ratio', 'vol_ewm', 'vol_lt'.
        """
        train = df.head(n_bars).dropna()
        if len(train) < 100:
            log.warning('AKF: insufficient data'); return

        log_px  = torch.tensor(
            np.log(train['close'].values), dtype=torch.float32
        ).to(DEVICE)

        # Context features per bar
        ctx_df = pd.DataFrame({
            'vol_ratio_norm': (train['vol_ratio'] / (train['vol_ratio'].rolling(60).mean() + 1e-9)).fillna(1.0),
            'ret_5_abs'     : train['close'].pct_change(5).abs().fillna(0.0),
            'vol_ewm_norm'  : (train.get('vol_ewm', pd.Series(0.2, index=train.index)) /
                               (train.get('vol_lt',  pd.Series(0.2, index=train.index)) + 1e-9)).fillna(1.0),
            'innov_abs'     : train.get('kf_innov', pd.Series(0.0, index=train.index)).abs().fillna(0.0),
        }).fillna(0.0).clip(-5, 5)

        ctx = torch.tensor(
            ctx_df.values, dtype=torch.float32
        ).to(DEVICE)                                         # (T, 4)

        T    = len(log_px)
        sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=AKF_EPOCHS)
        best_loss = np.inf

        for epoch in range(AKF_EPOCHS):
            self.opt.zero_grad()

            # Run differentiable Kalman through entire training sequence
            x = torch.zeros(1, 2, device=DEVICE)
            P = torch.eye(2, device=DEVICE).unsqueeze(0) * 1e4
            total_loss = torch.tensor(0.0, device=DEVICE)

            for t in range(T):
                c   = ctx[t:t+1]                         # (1,4)
                y   = log_px[t:t+1]                      # (1,)
                q_L, q_V, r = self.noise_net(c)
                x, P, nu, S = self._kf_step(x, P, y, q_L, q_V, r)
                # NLML contribution: ½(ν²/S + log S)
                total_loss = total_loss + 0.5 * (nu[0]**2 / S[0] + torch.log(S[0]))

            total_loss.backward()
            nn.utils.clip_grad_norm_(self.noise_net.parameters(), 1.0)
            self.opt.step()
            sched.step()

            loss_val = total_loss.item() / T
            if loss_val < best_loss:
                best_loss = loss_val
                self._best_state = {k: v.clone() for k, v in
                                    self.noise_net.state_dict().items()}
            if (epoch + 1) % 50 == 0:
                log.info(f'  AKF epoch {epoch+1}/{AKF_EPOCHS}  NLML/bar={loss_val:.4f}')

        self.noise_net.load_state_dict(self._best_state)
        self._trained = True
        log.info(f'  AKF trained — best NLML/bar={best_loss:.4f}')

    # ── inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def filter_series(self, df: pd.DataFrame
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run adaptive Kalman forward on full df.
        Returns (levels, velocities) as numpy arrays.
        If not trained, falls back to fixed defaults.
        """
        if not self._trained:
            # Fallback to fixed-noise KF (same as backtest)
            lp = np.log(df['close'].values.astype(float))
            from alphascout_backtest import KalmanFilter as _FixedKF
            kf = _FixedKF()
            lv, vel, _ = kf.filter_series(lp)
            return lv, vel

        log_px = torch.tensor(
            np.log(df['close'].values), dtype=torch.float32
        ).to(DEVICE)

        ctx_df = pd.DataFrame({
            'vol_ratio_norm': (df['vol_ratio'] / (df['vol_ratio'].rolling(60).mean() + 1e-9)).fillna(1.0),
            'ret_5_abs'     : df['close'].pct_change(5).abs().fillna(0.0),
            'vol_ewm_norm'  : (df.get('vol_ewm', pd.Series(0.2, index=df.index)) /
                               (df.get('vol_lt',  pd.Series(0.2, index=df.index)) + 1e-9)).fillna(1.0),
            'innov_abs'     : df.get('kf_innov', pd.Series(0.0, index=df.index)).abs().fillna(0.0),
        }).fillna(0.0).clip(-5, 5)
        ctx = torch.tensor(ctx_df.values, dtype=torch.float32).to(DEVICE)

        T  = len(log_px)
        x  = torch.zeros(1, 2, device=DEVICE)
        P  = torch.eye(2, device=DEVICE).unsqueeze(0) * 1e4
        levels, vels = np.empty(T), np.empty(T)

        self.noise_net.eval()
        for t in range(T):
            c = ctx[t:t+1]; y = log_px[t:t+1]
            q_L, q_V, r = self.noise_net(c)
            x, P, _, _  = self._kf_step(x, P, y, q_L, q_V, r)
            levels[t]   = x[0, 0].item()
            vels[t]     = x[0, 1].item()

        return levels, vels

    def save(self, path: str):
        torch.save({'state': self.noise_net.state_dict(),
                    'trained': self._trained}, path)
        log.info(f'AKF saved → {path}')

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.noise_net.load_state_dict(ckpt['state'])
        self._trained = ckpt['trained']
        log.info(f'AKF loaded ← {path}')


# ══════════════════════════════════════════════════════════════════════════════
#  ML PIPELINE  (orchestrates all three components)
# ══════════════════════════════════════════════════════════════════════════════

class MLPipeline:
    """
    Trains and wraps the three ML models.

    Workflow
    ────────
    1. ml = MLPipeline()
    2. ml.fit(features_dict)        ← call once after compute_features()
    3. ml.augment(features_dict)    ← adds ML columns to each ticker's DataFrame
       - 'xgb_prob'    : P(5d return > 0) from XGBoost
       - 'regime_prob' : P(bull) from LSTM
    4. ml.save(dir)   /  ml.load(dir)
    """

    def __init__(self, use_akf: bool = False):
        self.xgb_models: Dict[str, WalkForwardXGB] = {}
        self.lstm        = RegimeLSTM()
        self.akf_models: Dict[str, AdaptiveKalman]  = {}
        self.use_akf     = use_akf
        self._fitted     = False

    def fit(self, features: Dict[str, pd.DataFrame], benchmark: str = 'SPY'):
        """Full training pipeline."""
        log.info('━' * 52)
        log.info(' ML Pipeline — Training')
        log.info('━' * 52)

        # Build ML-specific features
        ml_feat = build_ml_features(features)

        # 1. XGBoost per ticker ───────────────────────────────────────────
        log.info('[1/3] Walk-forward XGBoost (GPU=%s)', XGB_DEVICE)
        for t, df in ml_feat.items():
            if t == benchmark:
                continue
            log.info(f'  {t} …')
            model = WalkForwardXGB()
            df['xgb_prob'] = model.fit_predict(df)
            ml_feat[t]     = df
            self.xgb_models[t] = model

        # 2. LSTM regime classifier on benchmark ──────────────────────────
        log.info('[2/3] LSTM Regime Classifier (device=%s)', DEVICE.type)
        spy_df = ml_feat.get(benchmark)
        if spy_df is not None:
            self.lstm.train(spy_df)
            regime_probs = self.lstm.predict_series(spy_df)
            # Broadcast to all tickers (market-wide signal)
            for t in ml_feat:
                ml_feat[t]['regime_prob'] = regime_probs.reindex(ml_feat[t].index).ffill()
        else:
            log.warning('SPY not found — LSTM regime disabled')

        # 3. Adaptive Kalman (optional, most expensive) ───────────────────
        if self.use_akf:
            log.info('[3/3] Adaptive Kalman Filter (device=%s)', DEVICE.type)
            for t, df in ml_feat.items():
                log.info(f'  AKF {t} …')
                akf = AdaptiveKalman()
                akf.train_on(df)
                lv, vel = akf.filter_series(df)
                df['kf_level_akf'] = lv
                df['kf_vel_akf']   = vel
                ml_feat[t]         = df
                self.akf_models[t] = akf
        else:
            log.info('[3/3] Adaptive Kalman — skipped  (set use_akf=True to enable)')

        self._ml_features = ml_feat
        self._fitted      = True
        log.info('ML Pipeline training complete.')

    def augment(self, features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Add ML columns to the features dict.
        If fit() was already called, returns the augmented version.
        Otherwise calls fit() first.
        """
        if not self._fitted:
            self.fit(features)
        return self._ml_features

    def score_ml(self, row: pd.Series) -> int:
        """
        Extra 0–3 points based on ML signals.
        Added on top of the base 0–9 score in alphascout_backtest.score_row().
        Combined score: 0–12.  Raise ENTRY_SCORE_MIN to 7 when using ML.
        """
        s    = 0
        prob = float(row.get('xgb_prob', 0.5))
        if prob >= 0.70:   s += 3    # strong ML conviction
        elif prob >= 0.60: s += 2
        elif prob >= 0.55: s += 1
        return s

    def save(self, directory: str = '.'):
        os.makedirs(directory, exist_ok=True)
        for t, model in self.xgb_models.items():
            model.save(os.path.join(directory, f'xgb_{t}.pkl'))
        self.lstm.save(os.path.join(directory, 'lstm_regime.pt'))
        for t, akf in self.akf_models.items():
            akf.save(os.path.join(directory, f'akf_{t}.pt'))
        log.info(f'All ML models saved → {directory}/')

    def load(self, directory: str = '.', tickers: Optional[List[str]] = None,
             benchmark: str = 'SPY'):
        lstm_path = os.path.join(directory, 'lstm_regime.pt')
        if os.path.exists(lstm_path):
            self.lstm.load(lstm_path)
        if tickers:
            for t in tickers:
                if t == benchmark:
                    continue
                xgb_path = os.path.join(directory, f'xgb_{t}.pkl')
                if os.path.exists(xgb_path):
                    m = WalkForwardXGB(); m.load(xgb_path)
                    self.xgb_models[t] = m
                akf_path = os.path.join(directory, f'akf_{t}.pt')
                if os.path.exists(akf_path):
                    a = AdaptiveKalman(); a.load(akf_path)
                    self.akf_models[t] = a
        self._fitted = True
        log.info(f'ML models loaded ← {directory}/')


# ══════════════════════════════════════════════════════════════════════════════
#  ML-AWARE SCORING (drop-in replacement for alphascout_backtest.score_row)
# ══════════════════════════════════════════════════════════════════════════════

def score_row_ml(row: pd.Series, pipeline: MLPipeline) -> int:
    """
    Extended scoring: base score (0–9) + ML bonus (0–3) = 0–12.
    Use ENTRY_SCORE_MIN = 7 when calling this.
    """
    # Base signals (identical to alphascout_backtest.score_row)
    s = 0
    if row.get('kf_vel',    0.0) > 2e-4:              s += 1
    if row.get('close',     0.0) > row.get('kf_price', 0.0): s += 1
    if row.get('vol_scalar', 1.0) >= 0.75:             s += 1
    ou_z = float(row.get('ou_z', 0.0))
    if -3.5 < ou_z < 1.5:                              s += 1
    if ou_z < -1.5 or 0.0 < ou_z < 1.0:               s += 1
    if row.get('sharpe_m', 0.0) > 0.5:                 s += 1
    if row.get('sharpe_m', 0.0) > 1.0:                 s += 1
    if row.get('macd', 0.0) > row.get('macd_sig', 0.0): s += 1
    if row.get('vol_ratio', 1.0) > 1.1:                s += 1
    # ML bonus
    s += pipeline.score_ml(row)
    return s


def regime_from_ml(pipeline: MLPipeline, spy_df: pd.DataFrame) -> float:
    """
    Use LSTM P(bull) to compute regime exposure scalar.
    Replaces the simple Kalman-velocity threshold in alphascout_backtest.
    """
    prob = pipeline.lstm.predict_prob(spy_df)
    return pipeline.lstm.regime_scalar(prob)
