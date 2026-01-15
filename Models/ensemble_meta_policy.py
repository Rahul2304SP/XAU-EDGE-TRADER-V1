#!/usr/bin/env python
"""
ensemble_meta_policy.py

Meta-Policy Ensemble for Trading Actions with Learned Exits
============================================================

This ensemble manages BOTH entries AND exits using learned policies:

ENTRY POLICY (MetaPolicyNet):
-----------------------------
A neural network classifier that outputs: BUY / SELL / NO_TRADE
- Only active when flat (no open position)
- Learns when to trust each base model and when to abstain

EXIT POLICY (ExitPolicyNet):
----------------------------
A binary classifier that outputs: HOLD / EXIT
- Only active when a position is open
- Learns optimal exit timing based on trade state and market context
- Does NOT close on entry signal flip (no reversal-based exits)

IMPORTANT: 
- p_trade_eff is NOT used anywhere. Only raw p_trade and p_dir.
- Base models are inference-only teachers; ensemble handles all MT5 orders.
- Exits are based on learned policy OR max holding time, NOT signal reversals.

Entry Policy Features:
----------------------
For each base model m:
  - p_trade_m: sigmoid(trade_logit) - raw probability of a trade opportunity
  - p_dir_m: sigmoid(dir_logit) - directional probability
Common context features:
  - p_trend, vol_30m, xau_spread
  - time features, correlation features, momentum features

Entry Label Rule:
  score_up = ret_up_atr - lambda_dd * mae_up_atr
  score_dn = ret_dn_atr - lambda_dd * mae_dn_atr
  If max(score_up, score_dn) < ENTRY_SCORE_THRESHOLD => NO_TRADE
  else BUY if score_up >= score_dn, else SELL

Exit Policy Features:
---------------------
- Base model outputs (p_trade, p_dir per model)
- Context (p_trend, vol_30m, xau_spread)
- Trade state: side, age_bars, unrealized_pnl_atr, mae_since_entry_atr, mfe_since_entry_atr

Exit Label Rule:
  For each bar t in a trade from entry i:
    score_exit(t) = realized_return_atr - lambda_dd * realized_mae_atr - spread_cost
  Label EXIT if:
    score_exit(t) >= max_future_score - label_margin AND score_exit(t) >= score_min
  Otherwise HOLD

Usage:
------
  # Train entry policy
  python ensemble_meta_policy.py --train

  # Train exit policy (requires entry policy)
  python ensemble_meta_policy.py --train_exit_policy

  # Run live (with both policies)
  python ensemble_meta_policy.py --live

  # Dry run (no actual orders)
  python ensemble_meta_policy.py --live --dry_run
"""

from __future__ import annotations

# Fix joblib hanging on Windows when detecting CPU cores (must be before sklearn imports)
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import argparse
import importlib.util
import json
import logging
import logging.handlers
import math
import os
import sys
import time
import threading
import datetime
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Generator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score,
    f1_score, precision_score, recall_score, precision_recall_fscore_support
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None


# =============================================================================
# Logging Infrastructure
# =============================================================================
class ComponentLogger:
    """Logger with component prefix tags for structured logging."""
    
    BOOT = "BOOT"
    IO = "IO"
    DATA = "DATA"
    FEAT = "FEAT"
    INFER = "INFER"
    META = "META"
    LABEL = "LABEL"
    TRAIN = "TRAIN"
    VAL = "VAL"
    EXEC = "EXEC"
    HEARTBEAT = "HEARTBEAT"
    PERF = "PERF"
    ERROR = "ERROR"
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def _log(self, level: int, component: str, msg: str, *args, **kwargs) -> None:
        self.logger.log(level, f"{component} | {msg}", *args, **kwargs)
    
    def debug(self, component: str, msg: str, *args, **kwargs) -> None:
        self._log(logging.DEBUG, component, msg, *args, **kwargs)
    
    def info(self, component: str, msg: str, *args, **kwargs) -> None:
        self._log(logging.INFO, component, msg, *args, **kwargs)
    
    def warning(self, component: str, msg: str, *args, **kwargs) -> None:
        self._log(logging.WARNING, component, msg, *args, **kwargs)
    
    def error(self, component: str, msg: str, *args, **kwargs) -> None:
        self._log(logging.ERROR, component, msg, *args, **kwargs)
    
    def exception(self, component: str, msg: str, *args, **kwargs) -> None:
        self._log(logging.ERROR, component, msg, *args, exc_info=True, **kwargs)


# Global component logger (initialized in setup_logging)
clog: Optional[ComponentLogger] = None


class UTCFormatter(logging.Formatter):
    """Formatter that uses UTC timestamps."""
    converter = time.gmtime
    
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
            s = f"{t},{int(record.msecs):03d}"
        return s


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> ComponentLogger:
    """
    Configure logging with console and rotating file handler.
    
    Format: timestamp | level | component | message
    """
    global clog
    
    # Parse log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("ensemble_meta")
    logger.setLevel(numeric_level)
    logger.handlers.clear()
    logger.propagate = False  # Don't duplicate to root logger
    
    # Format with UTC timestamps
    fmt = UTCFormatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    
    # Rotating file handler
    if log_file is None:
        log_file = LOGS_DIR / "ensemble_meta.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=20 * 1024 * 1024,  # 20 MB
        backupCount=10,
        encoding="utf-8"
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    
    clog = ComponentLogger(logger)
    return clog


@contextmanager
def timed_block(component: str, operation: str) -> Generator[dict, None, None]:
    """Context manager for timing operations and logging duration."""
    timing = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        timing["elapsed_ms"] = elapsed
        if clog:
            clog.debug(ComponentLogger.PERF, f"{operation} elapsed_ms={elapsed:.2f}")


def log_array_stats(arr: np.ndarray, name: str) -> str:
    """Generate summary stats string for a numpy array."""
    if len(arr) == 0:
        return f"{name}=[empty]"
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return f"{name}=[all non-finite]"
    finite_arr = arr[finite_mask]
    return f"{name}[min={finite_arr.min():.4f} mean={finite_arr.mean():.4f} max={finite_arr.max():.4f}]"


# MT5 return code meanings
MT5_RETCODE_MAP = {
    10004: "REQUOTE",
    10006: "REJECT",
    10007: "CANCEL",
    10008: "PLACED",
    10009: "DONE",
    10010: "DONE_PARTIAL",
    10011: "ERROR",
    10012: "TIMEOUT",
    10013: "INVALID",
    10014: "INVALID_VOLUME",
    10015: "INVALID_PRICE",
    10016: "INVALID_STOPS",
    10017: "TRADE_DISABLED",
    10018: "MARKET_CLOSED",
    10019: "NO_MONEY",
    10020: "PRICE_CHANGED",
    10021: "PRICE_OFF",
    10022: "INVALID_EXPIRATION",
    10023: "ORDER_CHANGED",
    10024: "TOO_MANY_REQUESTS",
    10025: "NO_CHANGES",
    10026: "AUTOTRADING_DISABLED",
    10027: "CLIENT_DISABLED",
    10028: "FROZEN",
    10029: "INVALID_FILL",
    10030: "CONNECTION",
}

# =============================================================================
# Path configuration
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Outputs" / "ensemble_meta"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOGS_DIR = OUTPUT_DIR / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "XAUUSD"
MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH", "")

# =============================================================================
# Default model specs (can be overridden via CLI)
# =============================================================================
DEFAULT_MODEL_SPECS = [
    {
        "name": "MR",
        "script_path": str(BASE_DIR / "mr_sltp_edge_kelly.py"),  # replace with your base model path
        "ckpt_path": "checkpoints/mr_edge_demo.pt",  # replace with your checkpoint
        "kind": "mr",
    },
    {
        "name": "MOM15",
        "script_path": str(BASE_DIR / "mr_sltp_edge_kelly.py"),  # placeholder; swap in your momentum model
        "ckpt_path": "checkpoints/mom15_demo.pt",
        "kind": "mom15",
    },
]

# =============================================================================
# =============================================================================
# Sequence length caps (override base model values to reduce memory/compute)
# Set to None to use base model's value, or set a number to cap it
# =============================================================================
MAX_SHORT_LEN = None    # None = use base model's value
MAX_MID_LEN = None      # None = use base model's value
MAX_LONG_LEN = None     # None = use full 1000 (like training)
MAX_SLOW_LEN = None     # None = use full 480 (like training)

# =============================================================================
# Meta-policy training configuration
# =============================================================================
META_HORIZON_BARS = 15
META_ENTRY_SCORE_THRESHOLD = 1.0  # Increased from 0.20 - higher = more NO_TRADE labels
META_LAMBDA_DD = 0.50  # drawdown penalty
META_EPOCHS = 100
META_BATCH = 256
META_LR = 1e-3
META_HIDDEN = 64
META_DROPOUT = 0.10
META_LOG_BATCH_EVERY = 50
META_EARLY_STOP_PATIENCE = 5
META_VAL_SPLIT = 0.20
META_LABEL_SMOOTHING = 0.05
InferenceBatch = 512
META_VAL_LOG_BATCH_EVERY = None  # None disables per-batch val logging

# TensorBoard logging frequency
TB_STEP_LOG_EVERY = 10        # Log step metrics every N optimizer steps
TB_HISTOGRAM_EVERY = 200      # Log histograms every N optimizer steps
TB_SPREAD_COST_ATR = 0.22     # Spread cost in ATR units for PnL proxy

# =============================================================================
# Exit policy configuration (learned exits)
# =============================================================================
EXIT_MAX_HOLD_BARS = 60       # Maximum holding period (force exit)
EXIT_LAMBDA_DD = 0.60         # Drawdown penalty for exit scoring
EXIT_LABEL_MARGIN = 0.10      # Margin below optimal for acceptable exit
EXIT_SCORE_MIN = 0.00         # Minimum score to label as EXIT
EXIT_PROB_THRESHOLD = 0.5    # p_exit threshold for live execution
EXIT_EPOCHS = 20              # Training epochs for exit policy
EXIT_BATCH = 128              # Batch size for exit policy training
EXIT_LR = 1e-3                # Learning rate
EXIT_HIDDEN = 64              # Hidden layer size
EXIT_DROPOUT = 0.10           # Dropout rate
EXIT_EARLY_STOP_PATIENCE = 5  # Early stopping patience
EXIT_VAL_SPLIT = 0.20         # Validation split

# Exit action labels
EXIT_HOLD = 0
EXIT_EXIT = 1
EXIT_NAMES = ["HOLD", "EXIT"]

# =============================================================================
# Action labels
# =============================================================================
ACTION_BUY = 0
ACTION_SELL = 1
ACTION_NO_TRADE = 2
ACTION_NAMES = ["BUY", "SELL", "NO_TRADE"]
NUM_ACTIONS = 3

# =============================================================================
# Live configuration
# =============================================================================
LIVE_MAX_LOTS = 0.01
LIVE_LOT_STEP = 0.01
LIVE_LOG_EVERY_BARS = 10
LIVE_WARMUP_BARS = 0
LIVE_FETCH_BARS = 10000  # Must be > max(LONG_LEN) + indicator lookback (~200)
LIVE_LOG_FILENAME = "meta_policy_live_log.csv"
LIVE_MIN_CONF = 0  # Require this confidence to take a BUY/SELL
LIVE_POLL_SECONDS = 5.0  # Live loop cadence

# =============================================================================
# Engineered p_trend (same logic as base models)
# =============================================================================
TREND_A = 0.35
TREND_B = 0.60
TREND_C = 0.25
TREND_D = 0.10
TREND_BIAS = -0.30


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _compute_p_trend_from_series(close: np.ndarray, atr_px: np.ndarray, idx: int) -> float:
    """Compute engineered p_trend feature at index idx."""
    if idx < 16:
        return 0.0
    atr = float(atr_px[idx]) if idx < len(atr_px) else 0.0
    if atr <= 0:
        return 0.0
    r1 = (close[idx] - close[idx - 1]) / atr
    r5 = (close[idx] - close[idx - 5]) / atr
    r15 = (close[idx] - close[idx - 15]) / atr
    acc = r5 - r15
    trend = (close[idx] - close[idx - 15]) / (15.0 * atr)
    net_abs = abs(close[idx] - close[idx - 15])
    path = np.sum(np.abs(np.diff(close[idx - 15 : idx + 1])))
    er = net_abs / max(path, 1e-12)
    er = float(max(0.0, min(1.0, er)))
    score = (
        TREND_A * abs(r15)
        + TREND_B * er
        + TREND_C * abs(trend)
        + TREND_D * max(0.0, abs(r5) - abs(r1))
        + TREND_BIAS
    )
    return float(_sigmoid(score))


# =============================================================================
# Module loading utilities
# =============================================================================
def _load_module(path: str, name: str):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_checkpoint_flexible(module: Any, path: Path, device: torch.device):
    """Load checkpoint with flexible state_dict matching."""
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as exc:
        logging.warning("Default torch.load failed (%s). Retrying with weights_only=False.", exc)
        ckpt = torch.load(path, map_location=device, weights_only=False)
    feature_cols = ckpt.get("feature_cols")
    norm_params = ckpt.get("norm_params", {})
    if not feature_cols:
        raise RuntimeError("Checkpoint missing feature_cols.")
    model = module.TrendMRModel(in_features=len(feature_cols), slow_in_features=len(feature_cols)).to(device)
    model_state = model.state_dict()
    ckpt_state = ckpt["model"]
    filtered_state = {}
    dropped = []
    for k, v in ckpt_state.items():
        if k not in model_state:
            dropped.append(k)
            continue
        if model_state[k].shape != v.shape:
            dropped.append(k)
            continue
        filtered_state[k] = v
    if dropped:
        logging.warning("Dropping %d incompatible keys from %s", len(dropped), path)
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing:
        logging.warning("Checkpoint missing keys (%s): %s", path, missing)
    if unexpected:
        logging.warning("Checkpoint unexpected keys (%s): %s", path, unexpected)
    model.eval()
    return model, feature_cols, norm_params


# =============================================================================
# Model outputs dataclass (NO p_trade_eff)
# =============================================================================
@dataclass
class ModelOutputs:
    """Per-model raw outputs: p_trade and p_dir only."""
    p_trade: np.ndarray  # sigmoid(trade_logit)
    p_dir: np.ndarray    # sigmoid(dir_logit)


# =============================================================================
# Model Adapter
# =============================================================================
class ModelAdapter:
    """Adapter for loading and running inference on a base model."""
    
    def __init__(self, spec: dict[str, Any]) -> None:
        self.name = spec["name"]
        self.kind = spec["kind"]
        self.script_path = spec["script_path"]
        
        if clog:
            clog.info(ComponentLogger.IO, f"Loading model module name={self.name} path={self.script_path}")
        
        self.module = _load_module(spec["script_path"], f"meta_ensemble_{self.name}")
        # Do NOT force slow stream on; respect the model's own setting
        if hasattr(self.module, "USE_SLOW_STREAM"):
            self.module.USE_SLOW_STREAM = bool(getattr(self.module, "USE_SLOW_STREAM", False))
        self.ckpt_path = Path(spec["ckpt_path"])
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required but not available. Cannot run on CPU.")
        self.device = torch.device("cuda")
        
        if clog:
            clog.info(ComponentLogger.IO, f"Loading checkpoint name={self.name} path={self.ckpt_path}")
        
        model, feature_cols, norm_params = _load_checkpoint_flexible(self.module, self.ckpt_path, self.device)
        self.model = model.to(self.device)
        self.model.eval()
        self.feature_cols = feature_cols
        self.norm_params = norm_params
        
        # Extract sequence lengths from module (with optional caps)
        self.short_len = getattr(self.module, "SHORT_LEN", 15)
        self.mid_len = getattr(self.module, "MID_LEN", 30)
        self.long_len = getattr(self.module, "LONG_LEN", 120)
        self.slow_len = getattr(self.module, "SLOW_LEN", 240)
        
        # Apply caps if configured (reduces memory usage for long sequences)
        orig_long, orig_slow = self.long_len, self.slow_len
        if MAX_SHORT_LEN is not None:
            self.short_len = min(self.short_len, MAX_SHORT_LEN)
        if MAX_MID_LEN is not None:
            self.mid_len = min(self.mid_len, MAX_MID_LEN)
        if MAX_LONG_LEN is not None:
            self.long_len = min(self.long_len, MAX_LONG_LEN)
        if MAX_SLOW_LEN is not None:
            self.slow_len = min(self.slow_len, MAX_SLOW_LEN)
        
        if clog and (self.long_len != orig_long or self.slow_len != orig_slow):
            clog.warning(
                ComponentLogger.INFER,
                f"Sequence lengths capped for {self.name}: long {orig_long}→{self.long_len}, slow {orig_slow}→{self.slow_len}"
            )
        
        if clog:
            clog.info(
                ComponentLogger.INFER,
                f"Model ready name={self.name} kind={self.kind} features={len(feature_cols)} "
                f"seq_lens=(S={self.short_len},M={self.mid_len},L={self.long_len},SLOW={self.slow_len}) "
                f"device={self.device} eval_mode={not self.model.training}"
            )

    def build_frame(self) -> tuple[np.ndarray, pd.DataFrame]:
        """Build feature frame for this model."""
        if clog:
            clog.info(ComponentLogger.FEAT, f"Building feature frame for model={self.name}")
        
        with timed_block(ComponentLogger.FEAT, f"build_frame_{self.name}") as timing:
            df, feature_cols = self.module._build_feature_frame()
            bar_dt = pd.to_datetime(df["bar_dt"], utc=True, errors="coerce")
            df = df.assign(bar_dt=bar_dt)
        
        # Verify all required feature columns exist
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            if clog:
                clog.warning(
                    ComponentLogger.FEAT, 
                    f"Missing {len(missing_cols)} columns for {self.name}: {missing_cols[:5]}..."
                )
            # Try to add regime features if that's what's missing
            if any("regime" in col.lower() for col in missing_cols):
                if hasattr(self.module, '_ensure_regime_params') and hasattr(self.module, '_add_regime_features'):
                    if clog:
                        clog.info(ComponentLogger.FEAT, f"Adding regime features for {self.name}")
                    params = self.module._ensure_regime_params(df, use_mt5=False)
                    df, _ = self.module._add_regime_features(df, list(feature_cols), params)
        
        if clog:
            # Log data summary
            start_ts = bar_dt.iloc[0] if len(bar_dt) > 0 else None
            end_ts = bar_dt.iloc[-1] if len(bar_dt) > 0 else None
            clog.info(
                ComponentLogger.DATA,
                f"Frame built model={self.name} rows={len(df)} cols={len(df.columns)} "
                f"start={start_ts} end={end_ts} elapsed_ms={timing['elapsed_ms']:.2f}"
            )
            # Log NaN counts
            nan_counts = df.isna().sum()
            top_nans = nan_counts[nan_counts > 0].nlargest(5)
            if len(top_nans) > 0:
                clog.debug(ComponentLogger.DATA, f"Top NaN columns: {top_nans.to_dict()}")
            # Log close price range
            if "xau_close" in df.columns:
                close = df["xau_close"]
                clog.debug(ComponentLogger.DATA, f"Close price min={close.min():.2f} max={close.max():.2f}")
            # Verify columns
            still_missing = [col for col in self.feature_cols if col not in df.columns]
            if still_missing:
                clog.error(ComponentLogger.ERROR, f"Still missing columns: {still_missing}")
        
        return bar_dt.to_numpy(), df

    def _norm_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Apply normalization to feature matrix."""
        return self.module._apply_norm_params(df, self.feature_cols, self.norm_params)

    def _build_seq_at(self, norm_matrix: np.ndarray, df: pd.DataFrame, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build sequences for a single bar index."""
        short = self.module._build_sequence(norm_matrix, idx, self.short_len)
        mid = self.module._build_sequence(norm_matrix, idx, self.mid_len)
        long = self.module._build_sequence(norm_matrix, idx, self.long_len)
        if getattr(self.module, "USE_SLOW_STREAM", False) and hasattr(self.module, "_build_slow_sequence_live"):
            slow = self.module._build_slow_sequence_live(df.iloc[: idx + 1], self.feature_cols)
        else:
            slow = np.zeros((self.slow_len, len(self.feature_cols)), dtype=np.float32)
        return short, mid, long, slow

    def _infer_single(self, short_t: torch.Tensor, mid_t: torch.Tensor, long_t: torch.Tensor, slow_t: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference and extract p_trade, p_dir (NO p_trade_eff).
        Returns shapes: (batch,), (batch,)
        """
        with torch.no_grad():
            if self.kind == "mr":
                # MR_SLTP_EXIT: returns (regime, trade, dir, bounce, hold, ksl, ktp)
                _, trade_logit, dir_logit, _, _, _, _ = self.model(short_t, mid_t, long_t, slow_t)
            elif self.kind == "regime":
                # MR_SLTP_EXIT_REGIME: returns (trade, dir, bounce, hold, ksl, ktp)
                trade_logit, dir_logit, _, _, _, _ = self.model(short_t, mid_t, long_t, slow_t)
            elif self.kind == "mom15":
                # 15_SL_TP_MOM_DYN: returns (regime, trade, dir, bounce, ksl, ktp, dist, break)
                _, trade_logit, dir_logit, _, _, _, _, _ = self.model(short_t, mid_t, long_t, slow_t)
            else:
                raise ValueError(f"Unknown model kind: {self.kind}")
            
            p_trade = torch.sigmoid(trade_logit).cpu().numpy()
            p_dir = torch.sigmoid(dir_logit).cpu().numpy()
        
        return p_trade.reshape(-1), p_dir.reshape(-1)

    def _build_sequences_on_gpu(self, norm_tensor: torch.Tensor, indices: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Build sequences on GPU using memory-efficient gather operation.
        
        Args:
            norm_tensor: (T, F) normalized feature matrix on GPU
            indices: (N,) end indices for each sequence on GPU
            seq_len: length of sequences to build
            
        Returns:
            (N, seq_len, F) tensor of sequences
        """
        n = indices.shape[0]
        T, F = norm_tensor.shape
        
        # Pre-allocate output tensor
        sequences = torch.zeros((n, seq_len, F), dtype=norm_tensor.dtype, device=self.device)
        
        # Build sequences one position at a time (memory efficient)
        for t in range(seq_len):
            # Offset from end: t=0 is -(seq_len-1), t=seq_len-1 is 0
            offset = t - (seq_len - 1)
            pos_indices = (indices + offset).clamp(0, T - 1)
            
            # Only copy where index is valid (not padding)
            valid_mask = (indices + offset) >= 0
            sequences[valid_mask, t, :] = norm_tensor[pos_indices[valid_mask]]
        
        return sequences

    def infer_batch(self, df: pd.DataFrame, indices: np.ndarray, batch_size: int = 128) -> ModelOutputs:
        """
        Run batched inference across given indices.
        Returns ModelOutputs with p_trade, p_dir only (no p_trade_eff).
        
        OPTIMIZED: Builds sequences on GPU in CHUNKS to avoid OOM.
        """
        # Process like training scripts - small batches, build sequences on-the-fly
        # This matches how individual model training works (batch_size 32-64)
        GPU_CHUNK_SIZE = 8192  # Same as typical training batch size
        THROTTLE_SECONDS = 0  # No throttle needed with small batches
        
        if clog:
            clog.info(ComponentLogger.INFER, f"Starting chunked GPU inference model={self.name} n={len(indices)} chunk={GPU_CHUNK_SIZE}")
        
        start_time = time.perf_counter()
        
        # Step 1: Normalize on CPU
        norm_start = time.perf_counter()
        norm_matrix = self._norm_matrix(df)
        norm_ms = (time.perf_counter() - norm_start) * 1000
        if clog:
            clog.info(ComponentLogger.PERF, f"Normalization elapsed_ms={norm_ms:.1f}")
        
        # Step 2: Upload feature matrix to GPU (small - just T x F)
        upload_start = time.perf_counter()
        norm_tensor = torch.from_numpy(norm_matrix).to(self.device)
        upload_ms = (time.perf_counter() - upload_start) * 1000
        if clog:
            clog.info(ComponentLogger.PERF, f"GPU upload elapsed_ms={upload_ms:.1f} shape={norm_matrix.shape}")
        
        # Step 3: Process in chunks to avoid OOM
        p_trade_list = []
        p_dir_list = []
        use_slow = getattr(self.module, "USE_SLOW_STREAM", False)
        total_chunks = math.ceil(len(indices) / GPU_CHUNK_SIZE)
        
        # Log GPU memory before starting
        if clog:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                clog.info(ComponentLogger.PERF, f"GPU memory before chunks: allocated={allocated:.2f}GB reserved={reserved:.2f}GB")
            clog.info(ComponentLogger.INFER, f"Processing {total_chunks} chunks of {GPU_CHUNK_SIZE}...")
        
        for chunk_num, chunk_start in enumerate(range(0, len(indices), GPU_CHUNK_SIZE), start=1):
            chunk_end = min(chunk_start + GPU_CHUNK_SIZE, len(indices))
            chunk_indices = indices[chunk_start:chunk_end]
            
            if clog and chunk_num == 1:
                clog.debug(ComponentLogger.INFER, f"Building chunk 1: {len(chunk_indices)} samples")
            
            chunk_indices_t = torch.from_numpy(chunk_indices.astype(np.int64)).to(self.device)
            
            # Build sequences for this chunk on GPU - with debug logging for first chunk
            if clog and chunk_num == 1:
                clog.debug(ComponentLogger.INFER, f"Building short sequences (len={self.short_len})...")
            short_chunk = self._build_sequences_on_gpu(norm_tensor, chunk_indices_t, self.short_len)
            
            if clog and chunk_num == 1:
                clog.debug(ComponentLogger.INFER, f"Building mid sequences (len={self.mid_len})...")
            mid_chunk = self._build_sequences_on_gpu(norm_tensor, chunk_indices_t, self.mid_len)
            
            if clog and chunk_num == 1:
                clog.debug(ComponentLogger.INFER, f"Building long sequences (len={self.long_len})...")
            long_chunk = self._build_sequences_on_gpu(norm_tensor, chunk_indices_t, self.long_len)
            
            if clog and chunk_num == 1:
                clog.debug(ComponentLogger.INFER, f"Building slow sequences (len={self.slow_len}, use_slow={use_slow})...")
            if use_slow:
                slow_chunk = self._build_sequences_on_gpu(norm_tensor, chunk_indices_t, self.slow_len)
            else:
                slow_chunk = torch.zeros((len(chunk_indices), self.slow_len, norm_matrix.shape[1]),
                                        dtype=torch.float32, device=self.device)
            
            if clog and chunk_num == 1:
                allocated = torch.cuda.memory_allocated() / 1e9
                clog.debug(ComponentLogger.INFER, f"Chunk 1 sequences built, GPU allocated={allocated:.2f}GB")
            
            # Run inference on this chunk
            for batch_start in range(0, len(chunk_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(chunk_indices))
                
                p_trade, p_dir = self._infer_single(
                    short_chunk[batch_start:batch_end],
                    mid_chunk[batch_start:batch_end],
                    long_chunk[batch_start:batch_end],
                    slow_chunk[batch_start:batch_end]
                )
                p_trade_list.append(p_trade)
                p_dir_list.append(p_dir)
            
            # Free chunk memory
            del short_chunk, mid_chunk, long_chunk, slow_chunk, chunk_indices_t
            torch.cuda.empty_cache()
            
            # Throttle to prevent system overload
            if THROTTLE_SECONDS > 0:
                time.sleep(THROTTLE_SECONDS)
            
            # Progress
            if clog:
                elapsed = time.perf_counter() - start_time
                pct = 100.0 * chunk_end / len(indices)
                rate = chunk_end / elapsed if elapsed > 0 else 0
                eta = (len(indices) - chunk_end) / rate if rate > 0 else 0
                clog.info(ComponentLogger.INFER, f"Chunk {chunk_num}/{total_chunks} ({pct:.0f}%) rate={rate:.0f}/s ETA={eta:.1f}s")
        
        # Cleanup
        del norm_tensor
        torch.cuda.empty_cache()
        
        p_trade_all = np.concatenate(p_trade_list)
        p_dir_all = np.concatenate(p_dir_list)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        throughput = len(indices) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        
        # Check for NaNs/Infs
        has_nan = np.any(~np.isfinite(p_trade_all)) or np.any(~np.isfinite(p_dir_all))
        
        if clog:
            clog.info(
                ComponentLogger.INFER,
                f"Inference done model={self.name} elapsed_ms={elapsed_ms:.0f} throughput={throughput:.0f}/s "
                f"{log_array_stats(p_trade_all, 'p_trade')} {log_array_stats(p_dir_all, 'p_dir')}"
            )
            if has_nan:
                clog.error(ComponentLogger.ERROR, f"NaN/Inf detected in inference output for model={self.name}!")
        
        if has_nan:
            raise ValueError(f"NaN/Inf detected in inference output for model={self.name}")
        
        return ModelOutputs(
            p_trade=p_trade_all,
            p_dir=p_dir_all,
        )


# =============================================================================
# Meta-Policy Network
# =============================================================================
class MetaPolicyNet(nn.Module):
    """
    Small classifier that outputs probabilities for BUY / SELL / NO_TRADE.
    Input: [p_trade_m1, p_dir_m1, ..., p_trade_mN, p_dir_mN, p_trend, vol_30m, xau_spread]
    Output: 3-class logits
    """
    
    def __init__(self, in_dim: int, hidden: int = META_HIDDEN, dropout: float = META_DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Exit Policy Network (learned exits)
# =============================================================================
class ExitPolicyNet(nn.Module):
    """
    Binary classifier that outputs p_exit_now in [0,1].
    
    Input features:
    - Base model outputs: p_trade, p_dir (and optional p_hold, p_bounce, etc.)
    - Context: p_trend, vol_30m, xau_spread
    - Trade state: side, age_bars, unrealized_pnl_atr, mae_since_entry_atr, mfe_since_entry_atr
    
    Output: single logit -> sigmoid for p_exit
    """
    
    def __init__(self, in_dim: int, hidden: int = EXIT_HIDDEN, dropout: float = EXIT_DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),  # Single output for binary classification
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logit (apply sigmoid externally for probability)."""
        return self.net(x).squeeze(-1)


# =============================================================================
# Outcome computation for labels
# =============================================================================
def _compute_outcomes(df: pd.DataFrame, horizon: int, lambda_dd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ret_up_atr, ret_dn_atr, mae_up_atr, mae_dn_atr for each bar.
    Returns: (ret_up, mae_up, ret_dn, mae_dn) arrays
    """
    close = df["xau_close"].to_numpy(dtype=np.float64)
    high = df["xau_high"].to_numpy(dtype=np.float64) if "xau_high" in df.columns else close.copy()
    low = df["xau_low"].to_numpy(dtype=np.float64) if "xau_low" in df.columns else close.copy()
    vol = df["vol_30m"].to_numpy(dtype=np.float64)
    atr = vol * close
    
    n = len(close)
    ret_up = np.zeros(n, dtype=np.float64)
    ret_dn = np.zeros(n, dtype=np.float64)
    mae_up = np.zeros(n, dtype=np.float64)
    mae_dn = np.zeros(n, dtype=np.float64)
    
    for i in range(n - horizon):
        if atr[i] <= 0:
            continue
        entry = close[i]
        fut_high = np.max(high[i + 1 : i + horizon + 1])
        fut_low = np.min(low[i + 1 : i + horizon + 1])
        eps = 1e-8
        ret_up[i] = (fut_high - entry) / (atr[i] + eps)
        ret_dn[i] = (entry - fut_low) / (atr[i] + eps)
        mae_up[i] = max(0.0, (entry - fut_low) / (atr[i] + eps))
        mae_dn[i] = max(0.0, (fut_high - entry) / (atr[i] + eps))
    
    return ret_up, mae_up, ret_dn, mae_dn


def _compute_labels(ret_up: np.ndarray, mae_up: np.ndarray, ret_dn: np.ndarray, mae_dn: np.ndarray,
                    entry_threshold: float, lambda_dd: float) -> np.ndarray:
    """
    Compute action labels based on score threshold.
    
    score_up = ret_up - lambda_dd * mae_up
    score_dn = ret_dn - lambda_dd * mae_dn
    
    If max(score_up, score_dn) < entry_threshold => NO_TRADE
    else BUY if score_up >= score_dn, else SELL
    """
    score_up = ret_up - lambda_dd * mae_up
    score_dn = ret_dn - lambda_dd * mae_dn
    max_score = np.maximum(score_up, score_dn)
    
    labels = np.full(len(ret_up), ACTION_NO_TRADE, dtype=np.int64)
    tradeable_mask = max_score >= entry_threshold
    buy_mask = tradeable_mask & (score_up >= score_dn)
    sell_mask = tradeable_mask & (score_up < score_dn)
    labels[buy_mask] = ACTION_BUY
    labels[sell_mask] = ACTION_SELL
    
    return labels


# =============================================================================
# Exit policy label generation (learned exits)
# =============================================================================
def _compute_exit_score(
    entry_idx: int,
    exit_idx: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    side: int,  # +1 for long, -1 for short
    lambda_dd: float,
    spread_cost_atr: float = TB_SPREAD_COST_ATR,
) -> float:
    """
    Compute exit score at exit_idx for a trade entered at entry_idx.
    
    score = realized_return_atr - lambda_dd * realized_mae_atr - spread_cost
    """
    eps = 1e-8
    entry_price = close[entry_idx]
    exit_price = close[exit_idx]
    entry_atr = atr[entry_idx] + eps
    
    if side > 0:  # Long
        realized_return_atr = (exit_price - entry_price) / entry_atr
        # MAE for long: worst drawdown from entry to exit
        min_low = np.min(low[entry_idx:exit_idx + 1])
        realized_mae_atr = max(0.0, (entry_price - min_low) / entry_atr)
    else:  # Short
        realized_return_atr = (entry_price - exit_price) / entry_atr
        # MAE for short: worst adverse move up from entry to exit
        max_high = np.max(high[entry_idx:exit_idx + 1])
        realized_mae_atr = max(0.0, (max_high - entry_price) / entry_atr)
    
    score = realized_return_atr - lambda_dd * realized_mae_atr - spread_cost_atr
    return score


EXIT_SAMPLES_CACHE_DIR = OUTPUT_DIR / "exit_policy" / "sample_cache"
EXIT_SAMPLES_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _generate_exit_training_samples(
    df: pd.DataFrame,
    entry_indices: np.ndarray,
    entry_sides: np.ndarray,  # +1 for BUY, -1 for SELL
    model_outputs_all: list[tuple[np.ndarray, np.ndarray]],  # List of (p_trade, p_dir) per model
    max_hold_bars: int = EXIT_MAX_HOLD_BARS,
    lambda_dd: float = EXIT_LAMBDA_DD,
    label_margin: float = EXIT_LABEL_MARGIN,
    score_min: float = EXIT_SCORE_MIN,
    use_cache: bool = True,
    batch_size: int = 10000,  # Process in batches to save memory
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate exit training samples for all entry points.
    Uses disk caching to avoid memory issues with large datasets.
    
    For each entry at index i with side s:
      - For each bar t from i to min(i + max_hold_bars, len-1):
        - Compute exit score at t
        - Label EXIT if score[t] >= max_future_score - margin AND score[t] >= score_min
        - Otherwise HOLD
    
    Returns:
        features: (N, feature_dim) array of exit features
        labels: (N,) array of {EXIT_HOLD=0, EXIT_EXIT=1}
        metadata: (N, 3) array of (entry_idx, exit_idx, side) for debugging
    """
    # Check for cached samples
    n_models = len(model_outputs_all)
    cache_key = f"exit_h{max_hold_bars}_dd{lambda_dd:.2f}_m{label_margin:.2f}_n{len(entry_indices)}"
    cache_features_path = EXIT_SAMPLES_CACHE_DIR / f"{cache_key}_features.npy"
    cache_labels_path = EXIT_SAMPLES_CACHE_DIR / f"{cache_key}_labels.npy"
    cache_meta_path = EXIT_SAMPLES_CACHE_DIR / f"{cache_key}_meta.npy"
    
    if use_cache and cache_features_path.exists() and cache_labels_path.exists():
        print(f"[EXIT SAMPLES] Loading cached samples from {EXIT_SAMPLES_CACHE_DIR}", flush=True)
        try:
            features = np.load(cache_features_path)
            labels = np.load(cache_labels_path)
            metadata = np.load(cache_meta_path) if cache_meta_path.exists() else np.zeros((len(labels), 3), dtype=np.int64)
            print(f"[EXIT SAMPLES] Loaded {len(features)} cached samples", flush=True)
            return features, labels, metadata
        except Exception as e:
            print(f"[EXIT SAMPLES] Cache load failed: {e}, regenerating...", flush=True)
    
    close = df["xau_close"].to_numpy(dtype=np.float64)
    high = df["xau_high"].to_numpy(dtype=np.float64) if "xau_high" in df.columns else close.copy()
    low = df["xau_low"].to_numpy(dtype=np.float64) if "xau_low" in df.columns else close.copy()
    vol = df["vol_30m"].to_numpy(dtype=np.float64)
    atr = vol * close + 1e-8
    spread = df["xau_spread"].to_numpy(dtype=np.float64) if "xau_spread" in df.columns else np.zeros_like(close)
    
    # Context features - precompute p_trend for all bars
    print(f"[EXIT SAMPLES] Computing p_trend for {len(close)} bars...", flush=True)
    p_trend_arr = np.zeros(len(close), dtype=np.float64)
    close_np = close.astype(np.float64)
    atr_np = atr.astype(np.float64)
    p_trend_start = time.perf_counter()
    for idx in range(16, len(close)):
        p_trend_arr[idx] = _compute_p_trend_from_series(close_np, atr_np, idx)
    print(f"[EXIT SAMPLES] p_trend computed in {time.perf_counter() - p_trend_start:.1f}s", flush=True)
    
    # Process in batches to avoid memory issues
    n_entries = len(entry_indices)
    n_batches = (n_entries + batch_size - 1) // batch_size
    
    # Use temporary files for batch outputs
    batch_files = []
    total_samples = 0
    start_time = time.perf_counter()
    
    print(f"[EXIT SAMPLES] Processing {n_entries} entries in {n_batches} batches of {batch_size}...", flush=True)
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_entries)
        batch_entries = entry_indices[batch_start:batch_end]
        batch_sides = entry_sides[batch_start:batch_end]
        
        batch_features = []
        batch_labels = []
        batch_metadata = []
        
        for i, (entry_idx, side) in enumerate(zip(batch_entries, batch_sides)):
            entry_price = close[entry_idx]
            entry_atr = atr[entry_idx]
            
            # Compute exit scores for all possible exit times
            max_exit_idx = min(entry_idx + max_hold_bars, len(close) - 1)
            exit_scores = []
            for t in range(entry_idx, max_exit_idx + 1):
                score = _compute_exit_score(entry_idx, t, close, high, low, atr, side, lambda_dd)
                exit_scores.append(score)
            
            if not exit_scores:
                continue
            
            exit_scores = np.array(exit_scores)
            
            # For each bar t in the trade, determine label
            for hold_idx, t in enumerate(range(entry_idx, max_exit_idx + 1)):
                current_price = close[t]
                age_bars = t - entry_idx
                
                # Unrealized PnL in ATR units
                if side > 0:  # Long
                    unrealized_pnl_atr = (current_price - entry_price) / entry_atr
                    min_low_so_far = np.min(low[entry_idx:t + 1])
                    mae_since_entry_atr = max(0.0, (entry_price - min_low_so_far) / entry_atr)
                    max_high_so_far = np.max(high[entry_idx:t + 1])
                    mfe_since_entry_atr = max(0.0, (max_high_so_far - entry_price) / entry_atr)
                else:  # Short
                    unrealized_pnl_atr = (entry_price - current_price) / entry_atr
                    max_high_so_far = np.max(high[entry_idx:t + 1])
                    mae_since_entry_atr = max(0.0, (max_high_so_far - entry_price) / entry_atr)
                    min_low_so_far = np.min(low[entry_idx:t + 1])
                    mfe_since_entry_atr = max(0.0, (entry_price - min_low_so_far) / entry_atr)
                
                # Build feature vector
                feat = []
                for p_trade_arr, p_dir_arr in model_outputs_all:
                    feat.append(float(p_trade_arr[t]) if t < len(p_trade_arr) else 0.0)
                    feat.append(float(p_dir_arr[t]) if t < len(p_dir_arr) else 0.5)
                feat.append(float(p_trend_arr[t]))
                feat.append(float(vol[t]))
                feat.append(float(spread[t]))
                feat.append(float(side))
                feat.append(float(age_bars) / float(max_hold_bars))
                feat.append(float(unrealized_pnl_atr))
                feat.append(float(mae_since_entry_atr))
                feat.append(float(mfe_since_entry_atr))
                
                batch_features.append(feat)
                
                # Determine label
                current_score = exit_scores[hold_idx]
                max_future_score = np.max(exit_scores[hold_idx:]) if hold_idx < len(exit_scores) else current_score
                
                if current_score >= max_future_score - label_margin and current_score >= score_min:
                    batch_labels.append(EXIT_EXIT)
                else:
                    batch_labels.append(EXIT_HOLD)
                
                batch_metadata.append([entry_idx, t, side])
        
        # Save batch to disk
        if batch_features:
            batch_file = EXIT_SAMPLES_CACHE_DIR / f"batch_{batch_idx}.npz"
            np.savez_compressed(
                batch_file,
                features=np.array(batch_features, dtype=np.float32),
                labels=np.array(batch_labels, dtype=np.int64),
                metadata=np.array(batch_metadata, dtype=np.int64),
            )
            batch_files.append(batch_file)
            total_samples += len(batch_features)
        
        # Progress logging
        elapsed = time.perf_counter() - start_time
        pct = 100.0 * (batch_idx + 1) / n_batches
        rate = (batch_idx + 1) / max(elapsed, 0.001)
        eta = (n_batches - batch_idx - 1) / max(rate, 0.001)
        print(f"[EXIT SAMPLES] Batch {batch_idx+1}/{n_batches} ({pct:.1f}%) total_samples={total_samples} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
        
        # Clear batch memory
        del batch_features, batch_labels, batch_metadata
        import gc
        gc.collect()
    
    # Merge all batches
    print(f"[EXIT SAMPLES] Merging {len(batch_files)} batch files...", flush=True)
    all_features = []
    all_labels = []
    all_metadata = []
    
    for batch_file in batch_files:
        with np.load(batch_file) as data:
            all_features.append(data["features"].copy())
            all_labels.append(data["labels"].copy())
            all_metadata.append(data["metadata"].copy())
        # File handle is now closed, safe to delete
        try:
            batch_file.unlink()
        except PermissionError:
            pass  # Ignore if still locked, cleanup later
    
    if not all_features:
        feat_dim = n_models * 2 + 3 + 5
        return np.zeros((0, feat_dim), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.int64)
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metadata = np.concatenate(all_metadata, axis=0)
    
    # Save to cache
    if use_cache:
        print(f"[EXIT SAMPLES] Saving {len(features)} samples to cache...", flush=True)
        np.save(cache_features_path, features)
        np.save(cache_labels_path, labels)
        np.save(cache_meta_path, metadata)
        print(f"[EXIT SAMPLES] Cache saved to {EXIT_SAMPLES_CACHE_DIR}", flush=True)
    
    return features, labels, metadata


def _compute_exit_trade_state(
    entry_price: float,
    entry_atr: float,
    current_price: float,
    side: int,
    prices_since_entry: np.ndarray,  # high/low prices since entry
    age_bars: int,
    max_hold_bars: int,
) -> tuple[float, float, float, float]:
    """
    Compute trade state features for exit decision.
    
    Returns: (unrealized_pnl_atr, mae_since_entry_atr, mfe_since_entry_atr, normalized_age)
    """
    eps = 1e-8
    entry_atr = max(entry_atr, eps)
    
    if side > 0:  # Long
        unrealized_pnl_atr = (current_price - entry_price) / entry_atr
        mae_since_entry_atr = max(0.0, (entry_price - np.min(prices_since_entry[:, 1])) / entry_atr) if len(prices_since_entry) > 0 else 0.0
        mfe_since_entry_atr = max(0.0, (np.max(prices_since_entry[:, 0]) - entry_price) / entry_atr) if len(prices_since_entry) > 0 else 0.0
    else:  # Short
        unrealized_pnl_atr = (entry_price - current_price) / entry_atr
        mae_since_entry_atr = max(0.0, (np.max(prices_since_entry[:, 0]) - entry_price) / entry_atr) if len(prices_since_entry) > 0 else 0.0
        mfe_since_entry_atr = max(0.0, (entry_price - np.min(prices_since_entry[:, 1])) / entry_atr) if len(prices_since_entry) > 0 else 0.0
    
    normalized_age = float(age_bars) / float(max(max_hold_bars, 1))
    
    return unrealized_pnl_atr, mae_since_entry_atr, mfe_since_entry_atr, normalized_age


# =============================================================================
# TensorBoard helper functions
# =============================================================================
def _compute_grad_norm(model: nn.Module) -> float:
    """Compute global L2 gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def _compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy of probability distribution."""
    eps = 1e-8
    return -torch.sum(probs * torch.log(probs + eps), dim=-1)


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error for multiclass classification."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)
    return ece


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str = "Confusion Matrix") -> plt.Figure:
    """Create a matplotlib figure of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def _fig_to_image(fig: plt.Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy array for TensorBoard."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close(fig)
    return img_array.transpose(2, 0, 1)  # HWC -> CHW


def _run_validation(
    meta_policy: nn.Module,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    class_weights_t: torch.Tensor,
    ret_up_val: np.ndarray,
    mae_up_val: np.ndarray,
    ret_dn_val: np.ndarray,
    mae_dn_val: np.ndarray,
    spread_val: np.ndarray,
    lambda_dd: float,
    device: torch.device,
    batch_size: int = 1024,
    log_every_batches: Optional[int] = None,
) -> dict[str, Any]:
    """Run full validation and compute all metrics."""
    meta_policy.eval()
    
    all_logits = []
    all_probs = []
    total_loss = 0.0
    n_batches = 0
    val_start_time = time.perf_counter()
    
    with torch.no_grad():
        for start in range(0, X_val_t.shape[0], batch_size):
            end = min(start + batch_size, X_val_t.shape[0])
            batch_x = X_val_t[start:end]
            batch_y = y_val_t[start:end]
            
            logits = meta_policy(batch_x)
            probs = F.softmax(logits, dim=-1)
            loss = F.cross_entropy(logits, batch_y, weight=class_weights_t)
            
            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            total_loss += loss.item()
            n_batches += 1
            
            if log_every_batches and clog and (n_batches % max(1, log_every_batches) == 0):
                elapsed = time.perf_counter() - val_start_time
                pct = 100.0 * end / X_val_t.shape[0]
                rate = end / max(elapsed, 1e-6)
                eta = (X_val_t.shape[0] - end) / max(rate, 1e-6)
                clog.info(
                    ComponentLogger.VAL,
                    f"Validation progress: batch={n_batches} samples={end}/{X_val_t.shape[0]} ({pct:.1f}%) "
                    f"avg_loss_so_far={total_loss / n_batches:.6f} rate={rate:.0f}/s ETA={eta:.1f}s",
                )
    
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    val_loss = total_loss / max(n_batches, 1)
    
    val_pred = np.argmax(all_probs, axis=1)
    val_true = y_val_t.cpu().numpy()
    
    # Basic metrics
    val_acc = float(np.mean(val_pred == val_true))
    val_balanced_acc = balanced_accuracy_score(val_true, val_pred)
    val_f1_macro = f1_score(val_true, val_pred, average='macro', zero_division=0)
    
    # Trade rate
    trade_mask_pred = val_pred != ACTION_NO_TRADE
    trade_mask_true = val_true != ACTION_NO_TRADE
    val_trade_rate_pred = float(np.mean(trade_mask_pred))
    val_trade_rate_true = float(np.mean(trade_mask_true))
    
    # F1 on trade only (BUY/SELL as positive classes)
    # Binary: trade vs no_trade
    binary_pred = (val_pred != ACTION_NO_TRADE).astype(int)
    binary_true = (val_true != ACTION_NO_TRADE).astype(int)
    val_precision_trade = precision_score(binary_true, binary_pred, zero_division=0)
    val_recall_trade = recall_score(binary_true, binary_pred, zero_division=0)
    val_f1_trade_only = f1_score(binary_true, binary_pred, zero_division=0)
    
    # Per-class precision/recall/F1
    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        val_true, val_pred, labels=[ACTION_BUY, ACTION_SELL, ACTION_NO_TRADE], zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(val_true, val_pred, labels=[ACTION_BUY, ACTION_SELL, ACTION_NO_TRADE])
    
    # PnL proxy metrics
    realized_scores = np.zeros(len(val_pred))
    ret_on_trades = []
    mae_on_trades = []
    spread_on_trades = []
    
    for i, pred in enumerate(val_pred):
        spread_cost = spread_val[i] * TB_SPREAD_COST_ATR  # spread in ATR units
        if pred == ACTION_BUY:
            score = ret_up_val[i] - lambda_dd * mae_up_val[i] - spread_cost
            realized_scores[i] = score
            ret_on_trades.append(ret_up_val[i])
            mae_on_trades.append(mae_up_val[i])
            spread_on_trades.append(spread_cost)
        elif pred == ACTION_SELL:
            score = ret_dn_val[i] - lambda_dd * mae_dn_val[i] - spread_cost
            realized_scores[i] = score
            ret_on_trades.append(ret_dn_val[i])
            mae_on_trades.append(mae_dn_val[i])
            spread_on_trades.append(spread_cost)
        else:
            realized_scores[i] = 0.0
    
    val_avg_score = float(np.mean(realized_scores))
    val_total_score = float(np.sum(realized_scores))
    
    if np.any(trade_mask_pred):
        val_avg_score_on_trades = float(np.mean(realized_scores[trade_mask_pred]))
        val_win_rate = float(np.mean(realized_scores[trade_mask_pred] > 0))
        val_avg_ret_on_trades = float(np.mean(ret_on_trades)) if ret_on_trades else 0.0
        val_avg_mae_on_trades = float(np.mean(mae_on_trades)) if mae_on_trades else 0.0
        val_avg_spread_on_trades = float(np.mean(spread_on_trades)) if spread_on_trades else 0.0
    else:
        val_avg_score_on_trades = 0.0
        val_win_rate = 0.0
        val_avg_ret_on_trades = 0.0
        val_avg_mae_on_trades = 0.0
        val_avg_spread_on_trades = 0.0
    
    # Calibration metrics
    confidences = np.max(all_probs, axis=1)
    val_mean_confidence = float(np.mean(confidences))
    
    # Accuracy at high confidence (>= 0.6)
    high_conf_mask = confidences >= 0.6
    if np.any(high_conf_mask):
        val_acc_at_high_conf = float(np.mean(val_pred[high_conf_mask] == val_true[high_conf_mask]))
        val_trade_rate_at_high_conf = float(np.mean(val_pred[high_conf_mask] != ACTION_NO_TRADE))
    else:
        val_acc_at_high_conf = 0.0
        val_trade_rate_at_high_conf = 0.0
    
    # NO_TRADE rate at low confidence (< 0.4)
    low_conf_mask = confidences < 0.4
    if np.any(low_conf_mask):
        val_no_trade_rate_at_low_conf = float(np.mean(val_pred[low_conf_mask] == ACTION_NO_TRADE))
    else:
        val_no_trade_rate_at_low_conf = 0.0
    
    # ECE
    val_ece = _compute_ece(all_probs, val_true)
    
    return {
        "loss": val_loss,
        "accuracy": val_acc,
        "balanced_accuracy": val_balanced_acc,
        "f1_macro": val_f1_macro,
        "f1_trade_only": val_f1_trade_only,
        "precision_trade": val_precision_trade,
        "recall_trade": val_recall_trade,
        "trade_rate_pred": val_trade_rate_pred,
        "trade_rate_true": val_trade_rate_true,
        "per_class_precision": per_class_prec,
        "per_class_recall": per_class_rec,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "logits": all_logits,
        "probs": all_probs,
        "predictions": val_pred,
        "true_labels": val_true,
        # PnL proxy
        "pnl_avg_score": val_avg_score,
        "pnl_avg_score_on_trades": val_avg_score_on_trades,
        "pnl_total_score": val_total_score,
        "pnl_win_rate": val_win_rate,
        "pnl_avg_ret_on_trades": val_avg_ret_on_trades,
        "pnl_avg_mae_on_trades": val_avg_mae_on_trades,
        "pnl_avg_spread_on_trades": val_avg_spread_on_trades,
        # Calibration
        "calib_mean_confidence": val_mean_confidence,
        "calib_acc_at_high_conf": val_acc_at_high_conf,
        "calib_trade_rate_at_high_conf": val_trade_rate_at_high_conf,
        "calib_no_trade_rate_at_low_conf": val_no_trade_rate_at_low_conf,
        "calib_ece": val_ece,
    }


def _log_validation_to_tb(writer: SummaryWriter, val_metrics: dict[str, Any], global_step: int, num_models: int) -> None:
    """Log all validation metrics to TensorBoard."""
    # Loss and basic metrics
    writer.add_scalar("loss/val_step", val_metrics["loss"], global_step)
    writer.add_scalar("metrics/val_accuracy", val_metrics["accuracy"], global_step)
    writer.add_scalar("metrics/val_balanced_accuracy", val_metrics["balanced_accuracy"], global_step)
    writer.add_scalar("metrics/val_f1_macro", val_metrics["f1_macro"], global_step)
    writer.add_scalar("metrics/val_f1_trade_only", val_metrics["f1_trade_only"], global_step)
    writer.add_scalar("metrics/val_precision_trade", val_metrics["precision_trade"], global_step)
    writer.add_scalar("metrics/val_recall_trade", val_metrics["recall_trade"], global_step)
    writer.add_scalar("metrics/val_trade_rate_pred", val_metrics["trade_rate_pred"], global_step)
    writer.add_scalar("metrics/val_trade_rate_true", val_metrics["trade_rate_true"], global_step)
    
    # Per-class metrics (order: BUY, SELL, NO_TRADE)
    class_names_ordered = ["buy", "sell", "no_trade"]
    for i, cls_name in enumerate(class_names_ordered):
        writer.add_scalar(f"metrics/val_precision_{cls_name}", val_metrics["per_class_precision"][i], global_step)
        writer.add_scalar(f"metrics/val_recall_{cls_name}", val_metrics["per_class_recall"][i], global_step)
        writer.add_scalar(f"metrics/val_f1_{cls_name}", val_metrics["per_class_f1"][i], global_step)
    
    # PnL proxy metrics
    writer.add_scalar("pnl_proxy/val_avg_score", val_metrics["pnl_avg_score"], global_step)
    writer.add_scalar("pnl_proxy/val_avg_score_on_trades", val_metrics["pnl_avg_score_on_trades"], global_step)
    writer.add_scalar("pnl_proxy/val_total_score", val_metrics["pnl_total_score"], global_step)
    writer.add_scalar("pnl_proxy/val_win_rate", val_metrics["pnl_win_rate"], global_step)
    writer.add_scalar("pnl_proxy/val_avg_ret_atr_on_trades", val_metrics["pnl_avg_ret_on_trades"], global_step)
    writer.add_scalar("pnl_proxy/val_avg_mae_atr_on_trades", val_metrics["pnl_avg_mae_on_trades"], global_step)
    writer.add_scalar("pnl_proxy/val_avg_spread_cost_on_trades", val_metrics["pnl_avg_spread_on_trades"], global_step)
    
    # Calibration metrics
    writer.add_scalar("calib/val_mean_confidence", val_metrics["calib_mean_confidence"], global_step)
    writer.add_scalar("calib/val_accuracy_at_conf_gt_0_6", val_metrics["calib_acc_at_high_conf"], global_step)
    writer.add_scalar("calib/val_trade_rate_at_conf_gt_0_6", val_metrics["calib_trade_rate_at_high_conf"], global_step)
    writer.add_scalar("calib/val_no_trade_rate_at_low_conf", val_metrics["calib_no_trade_rate_at_low_conf"], global_step)
    writer.add_scalar("calib/val_ece", val_metrics["calib_ece"], global_step)
    
    # Confusion matrix as image
    cm = val_metrics["confusion_matrix"]
    cm_fig = _plot_confusion_matrix(cm, ["BUY", "SELL", "NO_TRADE"], "Validation Confusion Matrix")
    cm_img = _fig_to_image(cm_fig)
    writer.add_image("cm/val", cm_img, global_step)


# =============================================================================
# Training function
# =============================================================================
def train_meta_policy(
    adapters: list[ModelAdapter],
    horizon: int = META_HORIZON_BARS,
    entry_threshold: float = META_ENTRY_SCORE_THRESHOLD,
    lambda_dd: float = META_LAMBDA_DD,
    val_log_every_batches: Optional[int] = META_VAL_LOG_BATCH_EVERY,
) -> Path:
    """Train the meta-policy classifier with comprehensive TensorBoard logging."""
    if clog:
        clog.info(ComponentLogger.TRAIN, "=" * 60)
        clog.info(ComponentLogger.TRAIN, "Starting Meta-Policy Training")
        clog.info(ComponentLogger.TRAIN, "=" * 60)
        clog.info(ComponentLogger.TRAIN, f"Horizon: {horizon} bars")
        clog.info(ComponentLogger.TRAIN, f"Entry score threshold: {entry_threshold:.3f}")
        clog.info(ComponentLogger.TRAIN, f"Lambda (drawdown penalty): {lambda_dd:.3f}")
        clog.info(ComponentLogger.TRAIN, f"Models: {[a.name for a in adapters]}")
        clog.info(ComponentLogger.TRAIN, f"Epochs: {META_EPOCHS}, Batch: {META_BATCH}, LR: {META_LR}")
    
    # Build feature frames for all adapters
    if clog:
        clog.info(ComponentLogger.FEAT, "Building feature frames for all models...")
    frames = []
    total_feat_time = 0
    for adapter in adapters:
        with timed_block(ComponentLogger.FEAT, f"build_frame_{adapter.name}") as timing:
            bar_dt, df = adapter.build_frame()
        total_feat_time += timing["elapsed_ms"]
        frames.append((bar_dt, df))
    
    if clog:
        clog.info(ComponentLogger.PERF, f"All feature frames built total_elapsed_ms={total_feat_time:.2f}")
    
    # Find common timestamps
    if clog:
        clog.info(ComponentLogger.DATA, "Aligning timestamps across models...")
    common_ts = set(frames[0][0])
    original_count = len(common_ts)
    for i, (bar_dt, _) in enumerate(frames[1:], start=1):
        before = len(common_ts)
        common_ts &= set(bar_dt)
        dropped = before - len(common_ts)
        if dropped > 0 and clog:
            clog.debug(ComponentLogger.DATA, f"Model {adapters[i].name}: dropped {dropped} non-overlapping bars")
    common_ts = np.array(sorted(common_ts))
    if common_ts.size == 0:
        if clog:
            clog.error(ComponentLogger.ERROR, "No overlapping timestamps across models!")
        raise RuntimeError("No overlapping timestamps across models.")
    
    if clog:
        dropped_total = original_count - len(common_ts)
        clog.info(
            ComponentLogger.DATA,
            f"Alignment complete: {len(common_ts)} common bars "
            f"(dropped {dropped_total} from first model)"
        )
    
    # Build index maps for alignment
    idx_maps = []
    for bar_dt, df in frames:
        ts_to_idx = {ts: i for i, ts in enumerate(bar_dt)}
        idx_maps.append(np.array([ts_to_idx[ts] for ts in common_ts], dtype=np.int64))
    
    # Use first model's df as base for outcomes
    base_df = frames[0][1].iloc[idx_maps[0]].reset_index(drop=True)
    
    # Compute outcomes and labels
    if clog:
        clog.info(ComponentLogger.LABEL, f"Computing outcomes horizon={horizon} lambda_dd={lambda_dd} threshold={entry_threshold}")
    
    with timed_block(ComponentLogger.LABEL, "compute_outcomes") as outcome_timing:
        ret_up, mae_up, ret_dn, mae_dn = _compute_outcomes(base_df, horizon, lambda_dd)
        labels = _compute_labels(ret_up, mae_up, ret_dn, mae_dn, entry_threshold, lambda_dd)
    
    # Log outcome distribution
    if clog:
        clog.info(
            ComponentLogger.LABEL,
            f"Outcome stats: {log_array_stats(ret_up, 'ret_up_atr')} {log_array_stats(ret_dn, 'ret_dn_atr')}"
        )
        clog.info(
            ComponentLogger.LABEL,
            f"Outcome stats: {log_array_stats(mae_up, 'mae_up_atr')} {log_array_stats(mae_dn, 'mae_dn_atr')}"
        )
    
    # Log label distribution
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    
    if clog:
        clog.info(ComponentLogger.LABEL, "Label distribution:")
        for action_id, count in label_dist.items():
            pct = 100.0 * count / len(labels)
            clog.info(ComponentLogger.LABEL, f"  {ACTION_NAMES[action_id]}: {count} ({pct:.2f}%)")
        
        # Warnings for imbalanced labels
        no_trade_pct = label_dist.get(ACTION_NO_TRADE, 0) / len(labels) * 100
        if no_trade_pct > 90:
            clog.warning(ComponentLogger.LABEL, f"NO_TRADE > 90% ({no_trade_pct:.1f}%) - consider adjusting threshold")
        
        buy_count = label_dist.get(ACTION_BUY, 0)
        sell_count = label_dist.get(ACTION_SELL, 0)
        if buy_count > 0 and sell_count > 0:
            ratio = max(buy_count, sell_count) / min(buy_count, sell_count)
            if ratio > 3:
                clog.warning(ComponentLogger.LABEL, f"BUY/SELL imbalanced ratio={ratio:.1f}")
        
        # Debug: log first few samples
        clog.debug(ComponentLogger.LABEL, "First 3 label samples:")
        for i in range(min(3, len(labels))):
            score_buy = ret_up[i] - lambda_dd * mae_up[i]
            score_sell = ret_dn[i] - lambda_dd * mae_dn[i]
            ts = common_ts[i] if i < len(common_ts) else "?"
            clog.debug(
                ComponentLogger.LABEL,
                f"  i={i} ts={ts} label={ACTION_NAMES[labels[i]]} score_buy={score_buy:.4f} score_sell={score_sell:.4f}"
            )
    
    # Run inference on all models
    if clog:
        clog.info(ComponentLogger.INFER, "Running inference on all base models...")
    outputs = []
    total_infer_time = 0
    for adapter, idxs, (_, df) in zip(adapters, idx_maps, frames, strict=True):
        outputs.append(adapter.infer_batch(df, idxs))
    
    # Compute p_trend for all bars
    close_arr = base_df["xau_close"].to_numpy(dtype=np.float64)
    vol_arr = base_df["vol_30m"].to_numpy(dtype=np.float64)
    atr_px = vol_arr * close_arr
    p_trend_vals = np.array([
        _compute_p_trend_from_series(close_arr, atr_px, i) for i in range(len(base_df))
    ], dtype=np.float32)
    
    # Build feature matrix
    # Features: [model outputs] + [time context] + [market context] + [correlation context]
    feats = []
    feat_names = []
    
    # 1. Model outputs (p_trade, p_dir for each model)
    for i, out in enumerate(outputs):
        feats.append(out.p_trade.astype(np.float32))
        feat_names.append(f"p_trade_{i}")
        feats.append(out.p_dir.astype(np.float32))
        feat_names.append(f"p_dir_{i}")
    
    # 2. Engineered trend
    feats.append(p_trend_vals)
    feat_names.append("p_trend")
    
    # 3. Basic context (always available)
    feats.append(base_df["vol_30m"].to_numpy(dtype=np.float32))
    feat_names.append("vol_30m")
    feats.append(base_df["xau_spread"].to_numpy(dtype=np.float32))
    feat_names.append("xau_spread")
    
    # 4. Time context (important for session-based patterns)
    time_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    for col in time_cols:
        if col in base_df.columns:
            feats.append(base_df[col].to_numpy(dtype=np.float32))
            feat_names.append(col)
    
    # 5. Correlation asset context (XAG, DXY)
    corr_cols = ["xag_ret_1", "dxy_ret_1", "gold_silver_ratio"]
    for col in corr_cols:
        if col in base_df.columns:
            arr = base_df[col].to_numpy(dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0)  # Handle NaNs
            feats.append(arr)
            feat_names.append(col)
    
    # 6. Volatility/momentum context
    context_cols = ["atr_pct", "rsi_14", "ret_15"]
    for col in context_cols:
        if col in base_df.columns:
            arr = base_df[col].to_numpy(dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            feats.append(arr)
            feat_names.append(col)
    
    X = np.stack(feats, axis=1)
    
    if clog:
        clog.info(ComponentLogger.FEAT, f"Meta-policy features ({len(feat_names)}): {feat_names}")
    
    # Store spread for PnL proxy computation
    spread_arr = base_df["xau_spread"].to_numpy(dtype=np.float32)
    
    in_dim = X.shape[1]
    num_models = len(adapters)
    logging.info("Feature matrix shape: %s (in_dim=%d)", X.shape, in_dim)
    
    # Mask out invalid labels (last `horizon` bars have no valid outcome)
    valid_mask = np.arange(len(labels)) < (len(labels) - horizon)
    X = X[valid_mask]
    labels = labels[valid_mask]
    ret_up = ret_up[valid_mask]
    mae_up = mae_up[valid_mask]
    ret_dn = ret_dn[valid_mask]
    mae_dn = mae_dn[valid_mask]
    spread_arr = spread_arr[valid_mask]
    logging.info("After masking invalid: %d samples", len(labels))
    
    # Train/val split (time-ordered)
    n_total = len(labels)
    n_val = max(1, int(n_total * META_VAL_SPLIT))
    n_train = n_total - n_val
    
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = labels[:n_train], labels[n_train:]
    ret_up_val, mae_up_val = ret_up[n_train:], mae_up[n_train:]
    ret_dn_val, mae_dn_val = ret_dn[n_train:], mae_dn[n_train:]
    spread_val = spread_arr[n_train:]
    
    if clog:
        clog.info(ComponentLogger.TRAIN, f"Train: {n_train} samples, Val: {n_val} samples")
    
    # Compute class weights (inverse frequency)
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    class_weights = np.ones(NUM_ACTIONS, dtype=np.float32)
    for cls, cnt in zip(train_unique, train_counts):
        class_weights[cls] = n_train / (NUM_ACTIONS * cnt)
    if clog:
        clog.info(ComponentLogger.TRAIN, f"Class weights: {class_weights.tolist()}")
    
    # Setup device and model (GPU required)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required but not available. Cannot run on CPU.")
    device = torch.device("cuda")
    if clog:
        clog.info(ComponentLogger.TRAIN, f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    meta_policy = MetaPolicyNet(in_dim).to(device)
    opt = torch.optim.Adam(meta_policy.parameters(), lr=META_LR)
    
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    class_weights_t = torch.from_numpy(class_weights).to(device)
    
    # TensorBoard setup with descriptive run name
    run_name = f"h{horizon}_thr{entry_threshold:.2f}_lam{lambda_dd:.2f}_{time.strftime('%Y%m%d_%H%M%S')}"
    tb_dir = OUTPUT_DIR / "tb" / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    logging.info("TensorBoard logs: %s", tb_dir)
    
    # Log hyperparameters as text
    hparams = {
        "horizon": horizon,
        "entry_threshold": entry_threshold,
        "lambda_dd": lambda_dd,
        "epochs": META_EPOCHS,
        "batch_size": META_BATCH,
        "lr": META_LR,
        "hidden": META_HIDDEN,
        "dropout": META_DROPOUT,
        "label_smoothing": META_LABEL_SMOOTHING,
        "val_split": META_VAL_SPLIT,
        "early_stop_patience": META_EARLY_STOP_PATIENCE,
        "num_models": num_models,
        "model_names": [a.name for a in adapters],
        "in_dim": in_dim,
        "n_train": n_train,
        "n_val": n_val,
        "class_weights": class_weights.tolist(),
        "label_distribution": {ACTION_NAMES[k]: int(v) for k, v in label_dist.items()},
    }
    writer.add_text("hparams", json.dumps(hparams, indent=2), global_step=0)
    
    # Training loop
    best_metric = -float('inf')  # Use pnl_avg_score_on_trades as selection metric
    best_epoch = 0
    patience_counter = 0
    best_state = None
    global_step = 0
    
    # Rolling timing stats
    step_times = deque(maxlen=100)
    fwd_times = deque(maxlen=100)
    bwd_times = deque(maxlen=100)
    
    if clog:
        clog.info(ComponentLogger.TRAIN, "Starting training loop...")
    
    meta_policy.train()
    for epoch in range(1, META_EPOCHS + 1):
        epoch_start = time.perf_counter()
        perm = torch.randperm(X_train_t.shape[0], device=device)
        total_batches = math.ceil(X_train_t.shape[0] / META_BATCH)
        epoch_loss = 0.0
        n_batches = 0
        
        if clog:
            clog.info(ComponentLogger.TRAIN, f"Epoch {epoch}/{META_EPOCHS} starting, batches={total_batches}")
        
        for batch_num, start in enumerate(range(0, X_train_t.shape[0], META_BATCH), start=1):
            step_start = time.perf_counter()
            
            idx = perm[start : start + META_BATCH]
            batch_x = X_train_t[idx]
            batch_y = y_train_t[idx]
            
            # Forward pass
            fwd_start = time.perf_counter()
            logits = meta_policy(batch_x)
            
            # Cross-entropy with class weights and optional label smoothing
            if META_LABEL_SMOOTHING > 0:
                n_classes = NUM_ACTIONS
                smooth_labels = torch.zeros_like(logits).scatter_(
                    1, batch_y.unsqueeze(1), 1.0
                )
                smooth_labels = (1.0 - META_LABEL_SMOOTHING) * smooth_labels + META_LABEL_SMOOTHING / n_classes
                log_probs = F.log_softmax(logits, dim=-1)
                loss = -torch.sum(smooth_labels * log_probs * class_weights_t.unsqueeze(0), dim=-1).mean()
            else:
                loss = F.cross_entropy(logits, batch_y, weight=class_weights_t)
            fwd_ms = (time.perf_counter() - fwd_start) * 1000
            fwd_times.append(fwd_ms)
            
            # Backward pass
            bwd_start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            bwd_ms = (time.perf_counter() - bwd_start) * 1000
            bwd_times.append(bwd_ms)
            
            # Compute grad norm before step
            grad_norm = _compute_grad_norm(meta_policy)
            
            opt.step()
            global_step += 1
            
            step_ms = (time.perf_counter() - step_start) * 1000
            step_times.append(step_ms)
            
            epoch_loss += loss.item()
            n_batches += 1
            
            # Compute batch stats for logging
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)
                
                # Label counts
                label_buy = (batch_y == ACTION_BUY).sum().item()
                label_sell = (batch_y == ACTION_SELL).sum().item()
                label_no = (batch_y == ACTION_NO_TRADE).sum().item()
                
                # Prediction counts
                pred_buy = (pred == ACTION_BUY).sum().item()
                pred_sell = (pred == ACTION_SELL).sum().item()
                pred_no = (pred == ACTION_NO_TRADE).sum().item()
            
            # Step-level verbose logging (every step or every N steps)
            if clog and global_step % 1 == 0:  # Every step
                current_lr = opt.param_groups[0]['lr']
                clog.debug(
                    ComponentLogger.TRAIN,
                    f"step={global_step} epoch={epoch} batch={batch_num}/{total_batches} "
                    f"loss={loss.item():.6f} lr={current_lr:.2e} grad_norm={grad_norm:.4f} "
                    f"labels(B/S/N)={label_buy}/{label_sell}/{label_no} "
                    f"preds(B/S/N)={pred_buy}/{pred_sell}/{pred_no} "
                    f"fwd_ms={fwd_ms:.2f} bwd_ms={bwd_ms:.2f} step_ms={step_ms:.2f}"
                )
            
            # Step-level logging
            if global_step % TB_STEP_LOG_EVERY == 0:
                probs = F.softmax(logits.detach(), dim=-1)
                entropy = _compute_entropy(probs).mean().item()
                pred = torch.argmax(probs, dim=-1)
                
                # Get current learning rate
                current_lr = opt.param_groups[0]['lr']
                
                writer.add_scalar("loss/train_step", loss.item(), global_step)
                writer.add_scalar("lr", current_lr, global_step)
                writer.add_scalar("grad_norm", grad_norm, global_step)
                writer.add_scalar("logits/entropy_train", entropy, global_step)
                
                # Mean probabilities per class
                writer.add_scalar("probs/mean_buy_train", probs[:, ACTION_BUY].mean().item(), global_step)
                writer.add_scalar("probs/mean_sell_train", probs[:, ACTION_SELL].mean().item(), global_step)
                writer.add_scalar("probs/mean_no_trade_train", probs[:, ACTION_NO_TRADE].mean().item(), global_step)
                
                # Trade rates
                batch_trade_rate = (pred != ACTION_NO_TRADE).float().mean().item()
                label_trade_rate = (batch_y != ACTION_NO_TRADE).float().mean().item()
                writer.add_scalar("data/batch_trade_rate_train", batch_trade_rate, global_step)
                writer.add_scalar("data/label_trade_rate_train", label_trade_rate, global_step)
            
            # Histogram logging (less frequent)
            if global_step % TB_HISTOGRAM_EVERY == 0:
                probs = F.softmax(logits.detach(), dim=-1)
                
                # Probability histograms
                writer.add_histogram("probs/buy_hist", probs[:, ACTION_BUY].cpu().numpy(), global_step)
                writer.add_histogram("probs/sell_hist", probs[:, ACTION_SELL].cpu().numpy(), global_step)
                writer.add_histogram("probs/no_trade_hist", probs[:, ACTION_NO_TRADE].cpu().numpy(), global_step)
                
                # Logit histograms
                writer.add_histogram("logits/buy_hist", logits[:, ACTION_BUY].detach().cpu().numpy(), global_step)
                writer.add_histogram("logits/sell_hist", logits[:, ACTION_SELL].detach().cpu().numpy(), global_step)
                writer.add_histogram("logits/no_trade_hist", logits[:, ACTION_NO_TRADE].detach().cpu().numpy(), global_step)
                
                # Feature histograms
                batch_x_np = batch_x.cpu().numpy()
                for m in range(num_models):
                    p_trade_idx = m * 2
                    p_dir_idx = m * 2 + 1
                    writer.add_histogram(f"features/p_trade_model{m}_hist", batch_x_np[:, p_trade_idx], global_step)
                    writer.add_histogram(f"features/p_dir_model{m}_hist", batch_x_np[:, p_dir_idx], global_step)
                
                # Context features
                p_trend_idx = num_models * 2
                vol_idx = num_models * 2 + 1
                spread_idx = num_models * 2 + 2
                writer.add_histogram("context/p_trend_hist", batch_x_np[:, p_trend_idx], global_step)
                writer.add_histogram("context/vol_30m_hist", batch_x_np[:, vol_idx], global_step)
                writer.add_histogram("context/spread_hist", batch_x_np[:, spread_idx], global_step)
            
            if META_LOG_BATCH_EVERY > 0 and (batch_num % META_LOG_BATCH_EVERY) == 0:
                # Compute rolling averages
                avg_step_ms = np.mean(step_times) if step_times else 0
                avg_fwd_ms = np.mean(fwd_times) if fwd_times else 0
                avg_bwd_ms = np.mean(bwd_times) if bwd_times else 0
                
                if clog:
                    clog.info(
                        ComponentLogger.TRAIN,
                        f"Epoch {epoch}/{META_EPOCHS} batch {batch_num}/{total_batches} "
                        f"loss={loss.item():.6f} grad_norm={grad_norm:.4f} "
                        f"avg_step_ms={avg_step_ms:.2f} avg_fwd_ms={avg_fwd_ms:.2f} avg_bwd_ms={avg_bwd_ms:.2f}"
                    )
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        epoch_elapsed = (time.perf_counter() - epoch_start) * 1000
        
        if clog:
            clog.info(ComponentLogger.PERF, f"Epoch {epoch} training elapsed_ms={epoch_elapsed:.2f}")
        
        # End of epoch: run full validation
        if clog:
            clog.info(ComponentLogger.VAL, f"Running validation epoch={epoch}...")
        val_start = time.perf_counter()
        val_metrics = _run_validation(
            meta_policy, X_val_t, y_val_t, class_weights_t,
            ret_up_val, mae_up_val, ret_dn_val, mae_dn_val, spread_val,
            lambda_dd, device, log_every_batches=val_log_every_batches
        )
        val_elapsed = (time.perf_counter() - val_start) * 1000
        
        # Log all validation metrics
        _log_validation_to_tb(writer, val_metrics, global_step, num_models)
        
        # Epoch-level logging
        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_metrics["loss"], epoch)
        writer.add_scalar("epoch/val_accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("epoch/val_pnl_score_on_trades", val_metrics["pnl_avg_score_on_trades"], epoch)
        
        # Model selection based on PnL proxy
        current_metric = val_metrics["pnl_avg_score_on_trades"]
        is_best = current_metric > best_metric
        
        writer.add_scalar("select/best_metric", best_metric if best_metric > -float('inf') else current_metric, global_step)
        writer.add_scalar("select/is_best", float(is_best), global_step)
        
        meta_policy.train()
        
        # Verbose validation logging
        if clog:
            clog.info(
                ComponentLogger.VAL,
                f"Epoch {epoch}/{META_EPOCHS} validation: loss={val_metrics['loss']:.6f} "
                f"acc={val_metrics['accuracy']:.4f} bal_acc={val_metrics['balanced_accuracy']:.4f} "
                f"f1_macro={val_metrics['f1_macro']:.4f} f1_trade={val_metrics['f1_trade_only']:.4f}"
            )
            clog.info(
                ComponentLogger.VAL,
                f"  trade_rate_pred={val_metrics['trade_rate_pred']:.4f} trade_rate_true={val_metrics['trade_rate_true']:.4f} "
                f"pnl_score={val_metrics['pnl_avg_score_on_trades']:.4f} win_rate={val_metrics['pnl_win_rate']:.4f}"
            )
            
            # Log confusion matrix
            cm = val_metrics["confusion_matrix"]
            clog.info(ComponentLogger.VAL, f"Confusion Matrix (BUY/SELL/NO_TRADE):")
            clog.info(ComponentLogger.VAL, f"  Pred→  BUY   SELL  NO_TR")
            clog.info(ComponentLogger.VAL, f"  BUY  {cm[0,0]:5d} {cm[0,1]:5d} {cm[0,2]:5d}")
            clog.info(ComponentLogger.VAL, f"  SELL {cm[1,0]:5d} {cm[1,1]:5d} {cm[1,2]:5d}")
            clog.info(ComponentLogger.VAL, f"  NO_TR{cm[2,0]:5d} {cm[2,1]:5d} {cm[2,2]:5d}")
            
            clog.info(ComponentLogger.PERF, f"Validation elapsed_ms={val_elapsed:.2f}")
            
            best_str = " [BEST]" if is_best else ""
            clog.info(
                ComponentLogger.TRAIN,
                f"Epoch {epoch} summary: train_loss={avg_train_loss:.6f} val_loss={val_metrics['loss']:.6f} "
                f"val_acc={val_metrics['accuracy']:.4f} pnl_score={current_metric:.4f}{best_str}"
            )
        
        # Early stopping based on PnL metric
        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            best_state = meta_policy.state_dict().copy()
            if clog:
                clog.info(ComponentLogger.TRAIN, f"New best model at epoch {epoch} metric={best_metric:.4f}")
        else:
            patience_counter += 1
            if clog:
                clog.debug(ComponentLogger.TRAIN, f"No improvement, patience={patience_counter}/{META_EARLY_STOP_PATIENCE}")
            if patience_counter >= META_EARLY_STOP_PATIENCE:
                if clog:
                    clog.info(
                        ComponentLogger.TRAIN,
                        f"Early stopping at epoch {epoch} (best epoch={best_epoch}, best_metric={best_metric:.4f})"
                    )
                break
        
        writer.flush()
    
    # Load best state
    if best_state is not None:
        meta_policy.load_state_dict(best_state)
        if clog:
            clog.info(ComponentLogger.TRAIN, f"Loaded best model from epoch {best_epoch} (best_metric={best_metric:.4f})")
    
    # Final validation metrics
    if clog:
        clog.info(ComponentLogger.VAL, "Running final validation on best model...")
    final_val_metrics = _run_validation(
        meta_policy, X_val_t, y_val_t, class_weights_t,
        ret_up_val, mae_up_val, ret_dn_val, mae_dn_val, spread_val,
        lambda_dd, device
    )
    
    if clog:
        clog.info(ComponentLogger.TRAIN, "=" * 60)
        clog.info(ComponentLogger.TRAIN, "Final Validation Metrics (Best Model)")
        clog.info(ComponentLogger.TRAIN, "=" * 60)
        clog.info(ComponentLogger.VAL, f"Accuracy: {final_val_metrics['accuracy']:.4f}")
        clog.info(ComponentLogger.VAL, f"Balanced Accuracy: {final_val_metrics['balanced_accuracy']:.4f}")
        clog.info(ComponentLogger.VAL, f"F1 Macro: {final_val_metrics['f1_macro']:.4f}")
        clog.info(ComponentLogger.VAL, f"F1 Trade-Only: {final_val_metrics['f1_trade_only']:.4f}")
        clog.info(ComponentLogger.VAL, f"Trade Rate (pred): {final_val_metrics['trade_rate_pred']:.4f}")
        clog.info(ComponentLogger.VAL, f"Trade Rate (true): {final_val_metrics['trade_rate_true']:.4f}")
        clog.info(ComponentLogger.VAL, f"PnL Avg Score: {final_val_metrics['pnl_avg_score']:.4f}")
        clog.info(ComponentLogger.VAL, f"PnL Avg Score on Trades: {final_val_metrics['pnl_avg_score_on_trades']:.4f}")
        clog.info(ComponentLogger.VAL, f"PnL Win Rate: {final_val_metrics['pnl_win_rate']:.4f}")
        clog.info(ComponentLogger.VAL, f"Calibration ECE: {final_val_metrics['calib_ece']:.4f}")
        
        cm = final_val_metrics["confusion_matrix"]
        clog.info(ComponentLogger.VAL, "Final Confusion Matrix:")
        clog.info(ComponentLogger.VAL, f"  Pred→  BUY   SELL  NO_TR")
        clog.info(ComponentLogger.VAL, f"  BUY  {cm[0,0]:5d} {cm[0,1]:5d} {cm[0,2]:5d}")
        clog.info(ComponentLogger.VAL, f"  SELL {cm[1,0]:5d} {cm[1,1]:5d} {cm[1,2]:5d}")
        clog.info(ComponentLogger.VAL, f"  NO_TR{cm[2,0]:5d} {cm[2,1]:5d} {cm[2,2]:5d}")
    
    # Also log full classification report
    report = classification_report(
        final_val_metrics["true_labels"], 
        final_val_metrics["predictions"], 
        target_names=ACTION_NAMES, 
        zero_division=0
    )
    if clog:
        clog.info(ComponentLogger.VAL, f"Classification Report:\n{report}")
    
    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"meta_epoch{best_epoch}.pt"
    model_names = [a.name for a in adapters]
    feature_schema = {
        "num_models": len(adapters),
        "model_names": model_names,
        "per_model_features": ["p_trade", "p_dir"],
        "context_features": feat_names[len(adapters)*2:],  # All features after model outputs
        "all_features": feat_names,
        "in_dim": in_dim,
    }
    
    if clog:
        clog.info(ComponentLogger.IO, f"Saving checkpoint to {ckpt_path}")
    
    torch.save({
        "state_dict": meta_policy.state_dict(),
        "in_dim": in_dim,
        "feature_schema": feature_schema,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "final_val_metrics": {
            "accuracy": final_val_metrics["accuracy"],
            "balanced_accuracy": final_val_metrics["balanced_accuracy"],
            "f1_macro": final_val_metrics["f1_macro"],
            "pnl_avg_score_on_trades": final_val_metrics["pnl_avg_score_on_trades"],
            "pnl_win_rate": final_val_metrics["pnl_win_rate"],
        },
        "horizon": horizon,
        "entry_threshold": entry_threshold,
        "lambda_dd": lambda_dd,
    }, ckpt_path)
    
    if clog:
        clog.info(ComponentLogger.IO, f"Checkpoint saved: {ckpt_path}")
    
    # Also save schema JSON
    schema_path = CHECKPOINT_DIR / f"meta_epoch{best_epoch}_schema.json"
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump({
            "feature_schema": feature_schema,
            "model_specs": [{"name": s["name"], "kind": s["kind"]} for s in DEFAULT_MODEL_SPECS[:len(adapters)]],
            "horizon": horizon,
            "entry_threshold": entry_threshold,
            "lambda_dd": lambda_dd,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "tb_run": run_name,
        }, f, indent=2)
    
    if clog:
        clog.info(ComponentLogger.IO, f"Schema saved: {schema_path}")
        clog.info(ComponentLogger.TRAIN, "=" * 60)
        clog.info(ComponentLogger.TRAIN, "Training complete!")
        clog.info(ComponentLogger.TRAIN, "=" * 60)
    
    writer.flush()
    writer.close()
    
    return ckpt_path


# =============================================================================
# Live runner
# =============================================================================
def _mt5_initialize() -> bool:
    if mt5 is None:
        return False
    if MT5_TERMINAL_PATH:
        return bool(mt5.initialize(path=MT5_TERMINAL_PATH))
    return bool(mt5.initialize())


def _get_tick(symbol: str):
    if mt5 is None:
        return None
    return mt5.symbol_info_tick(symbol)


def _round_lot(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(x / step) * step


# =============================================================================
# Exit Policy Training
# =============================================================================
EXIT_CHECKPOINT_DIR = OUTPUT_DIR / "exit_policy" / "checkpoints"
EXIT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def train_exit_policy(
    adapters: list[ModelAdapter],
    entry_labels: np.ndarray,  # Entry labels from entry policy training
    max_hold_bars: int = EXIT_MAX_HOLD_BARS,
    lambda_dd: float = EXIT_LAMBDA_DD,
    label_margin: float = EXIT_LABEL_MARGIN,
    score_min: float = EXIT_SCORE_MIN,
    train_on_entry_policy: bool = True,
    samples_limit: Optional[int] = None,
    log_every_batches: Optional[int] = None,
    val_entries_limit: Optional[int] = None,
) -> Path:
    """
    Train the exit policy network.
    
    Args:
        adapters: List of model adapters for inference
        entry_labels: Entry labels (ACTION_BUY, ACTION_SELL, ACTION_NO_TRADE) from entry training
        max_hold_bars: Maximum holding period
        lambda_dd: Drawdown penalty for exit scoring
        label_margin: Margin below optimal for acceptable exit
        score_min: Minimum score to label as EXIT
        train_on_entry_policy: If True, only train on entries where entry policy would trade
        samples_limit: Optional limit on training samples
        log_every_batches: Optional override for how often to print batch progress per epoch
        val_entries_limit: Optional cap on the number of unique entries used for exit validation simulation
    
    Returns:
        Path to saved checkpoint
    """
    if clog:
        clog.info(ComponentLogger.BOOT, "=" * 70)
        clog.info(ComponentLogger.BOOT, "EXIT POLICY TRAINING")
        clog.info(ComponentLogger.BOOT, "=" * 70)
        clog.info(ComponentLogger.BOOT, f"max_hold_bars={max_hold_bars} lambda_dd={lambda_dd}")
        clog.info(ComponentLogger.BOOT, f"label_margin={label_margin} score_min={score_min}")
    
    device = torch.device("cuda")
    
    # TensorBoard setup
    run_name = f"exit_h{max_hold_bars}_dd{lambda_dd:.2f}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    tb_dir = OUTPUT_DIR / "exit_policy" / "tb" / run_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    # Build feature frames for all adapters (same as entry training)
    if clog:
        clog.info(ComponentLogger.FEAT, "Building feature frames for all models...")
    frames = []
    for adapter in adapters:
        bar_dt, adapter_df = adapter.build_frame()
        frames.append((bar_dt, adapter_df))
    
    # Find common timestamps across all models
    if clog:
        clog.info(ComponentLogger.DATA, "Aligning timestamps across models...")
    common_ts = set(frames[0][0])
    for bar_dt, _ in frames[1:]:
        common_ts &= set(bar_dt)
    common_ts = np.array(sorted(common_ts))
    
    if common_ts.size == 0:
        raise RuntimeError("No overlapping timestamps across models.")
    
    # Apply data limit (use only last N bars for faster testing)
    if samples_limit is not None and samples_limit < len(common_ts):
        # Use last N*2 bars to have enough room for training samples
        data_limit = min(len(common_ts), samples_limit * 3)
        if clog:
            clog.info(ComponentLogger.DATA, f"Limiting data to last {data_limit} bars (for faster testing)")
        common_ts = common_ts[-data_limit:]
    
    if clog:
        clog.info(ComponentLogger.DATA, f"Alignment complete: {len(common_ts)} common bars")
    
    # Build index maps for alignment
    idx_maps = []
    for bar_dt, _ in frames:
        ts_to_idx = {ts: i for i, ts in enumerate(bar_dt)}
        idx_maps.append(np.array([ts_to_idx[ts] for ts in common_ts], dtype=np.int64))
    
    # Use first model's df as base for outcomes
    df = frames[0][1].iloc[idx_maps[0]].reset_index(drop=True)
    
    if clog:
        clog.info(ComponentLogger.DATA, f"Base feature frame: {len(df)} rows")
    
    # Ensure high/low columns exist
    if "xau_high" not in df.columns:
        if "high" in df.columns:
            df["xau_high"] = df["high"]
        else:
            df["xau_high"] = df["xau_close"]
    if "xau_low" not in df.columns:
        if "low" in df.columns:
            df["xau_low"] = df["low"]
        else:
            df["xau_low"] = df["xau_close"]
    
    # Determine entry points for training FIRST (to check cache)
    max_valid_entry = len(df) - max_hold_bars - 1
    
    if train_on_entry_policy:
        entry_mask = (entry_labels == ACTION_BUY) | (entry_labels == ACTION_SELL)
        entry_indices = np.where(entry_mask)[0]
        valid_mask = entry_indices < max_valid_entry
        entry_indices = entry_indices[valid_mask]
        entry_sides = np.where(entry_labels[entry_indices] == ACTION_BUY, 1, -1)
    else:
        entry_indices = np.arange(max(0, max_valid_entry))
        entry_sides = np.ones(len(entry_indices), dtype=np.int64)
    
    if samples_limit is not None and len(entry_indices) > samples_limit:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(entry_indices), size=samples_limit, replace=False)
        entry_indices = entry_indices[sample_idx]
        entry_sides = entry_sides[sample_idx]
    
    if clog:
        clog.info(ComponentLogger.LABEL, f"Valid entry points: {len(entry_indices)} (max_valid_entry={max_valid_entry})")
    
    # Check if cache exists BEFORE running expensive inference
    n_models = len(adapters)
    cache_key = f"exit_h{max_hold_bars}_dd{lambda_dd:.2f}_m{label_margin:.2f}_n{len(entry_indices)}"
    cache_features_path = EXIT_SAMPLES_CACHE_DIR / f"{cache_key}_features.npy"
    cache_labels_path = EXIT_SAMPLES_CACHE_DIR / f"{cache_key}_labels.npy"
    cache_meta_path = EXIT_SAMPLES_CACHE_DIR / f"{cache_key}_meta.npy"
    
    if cache_features_path.exists() and cache_labels_path.exists():
        if clog:
            clog.info(ComponentLogger.IO, f"Loading cached exit samples (skipping inference)...")
        features = np.load(cache_features_path)
        labels = np.load(cache_labels_path)
        metadata = np.load(cache_meta_path) if cache_meta_path.exists() else np.zeros((len(labels), 3), dtype=np.int64)
        if clog:
            clog.info(ComponentLogger.IO, f"Loaded {len(features)} cached samples")
    else:
        # No cache - need to run inference
        if clog:
            clog.info(ComponentLogger.INFER, "Running inference on all adapters for exit training...")
        
        model_outputs_all = []
        for i, (adapter, idxs, (_, adapter_df)) in enumerate(zip(adapters, idx_maps, frames)):
            if clog:
                clog.info(ComponentLogger.INFER, f"Running inference for adapter {i+1}/{len(adapters)}: {adapter.name}")
            
            outputs = adapter.infer_batch(adapter_df, idxs)
            p_trade = outputs.p_trade
            p_dir = outputs.p_dir
            model_outputs_all.append((p_trade, p_dir))
            if clog:
                clog.info(ComponentLogger.INFER, f"  {adapter.name}: p_trade mean={np.mean(p_trade):.3f} p_dir mean={np.mean(p_dir):.3f}")
        
        if clog:
            clog.info(ComponentLogger.LABEL, f"Generating exit training samples for {len(entry_indices)} entries...")
        
        # Generate exit training samples
        features, labels, metadata = _generate_exit_training_samples(
            df, entry_indices, entry_sides, model_outputs_all,
            max_hold_bars=max_hold_bars, lambda_dd=lambda_dd,
            label_margin=label_margin, score_min=score_min,
        )
    
    if len(features) == 0:
        if clog:
            clog.error(ComponentLogger.ERROR, "No exit training samples generated!")
        raise RuntimeError("No exit training samples generated")
    
    if clog:
        clog.info(ComponentLogger.LABEL, f"Generated {len(features)} exit samples")
        exit_rate = np.mean(labels == EXIT_EXIT)
        clog.info(ComponentLogger.LABEL, f"Label distribution: HOLD={np.sum(labels == EXIT_HOLD)} EXIT={np.sum(labels == EXIT_EXIT)} ({exit_rate*100:.1f}% EXIT)")
    
    # Time-based train/val split
    n_samples = len(features)
    split_idx = int(n_samples * (1 - EXIT_VAL_SPLIT))
    
    train_X = torch.tensor(features[:split_idx], dtype=torch.float32, device=device)
    train_y = torch.tensor(labels[:split_idx], dtype=torch.float32, device=device)
    val_X = torch.tensor(features[split_idx:], dtype=torch.float32, device=device)
    val_y = torch.tensor(labels[split_idx:], dtype=torch.float32, device=device)
    val_metadata = metadata[split_idx:]  # Keep as numpy for PnL computation
    
    if clog:
        clog.info(ComponentLogger.TRAIN, f"Train samples: {len(train_X)}, Val samples: {len(val_X)}")
    
    # Initialize model
    in_dim = features.shape[1]
    model = ExitPolicyNet(in_dim=in_dim, hidden=EXIT_HIDDEN, dropout=EXIT_DROPOUT).to(device)
    
    # Compute class weights for imbalanced data
    pos_weight = torch.tensor([(labels == EXIT_HOLD).sum() / max((labels == EXIT_EXIT).sum(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=EXIT_LR)
    
    if clog:
        clog.info(ComponentLogger.TRAIN, f"Model input dim: {in_dim}, pos_weight: {pos_weight.item():.2f}")
    
    # Training loop
    best_val_metric = -float('inf')
    best_epoch = 0
    patience_counter = 0
    best_ckpt_path = EXIT_CHECKPOINT_DIR / "exit_policy_best.pt"
    
    global_step = 0
    
    for epoch in range(EXIT_EPOCHS):
        model.train()
        
        # Shuffle training data
        perm = torch.randperm(len(train_X))
        train_X_shuffled = train_X[perm]
        train_y_shuffled = train_y[perm]
        
        epoch_loss = 0.0
        n_batches = 0
        total_batches = (len(train_X_shuffled) + EXIT_BATCH - 1) // EXIT_BATCH
        epoch_start_time = time.perf_counter()
        if log_every_batches is None:
            current_log_every = max(1, total_batches // 10)  # Log ~10 times per epoch
        else:
            current_log_every = max(1, min(total_batches, log_every_batches))
        
        for batch_start in range(0, len(train_X_shuffled), EXIT_BATCH):
            batch_X = train_X_shuffled[batch_start:batch_start + EXIT_BATCH]
            batch_y = train_y_shuffled[batch_start:batch_start + EXIT_BATCH]
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            grad_norm = _compute_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            
            # Progress logging within epoch
            if n_batches % current_log_every == 0:
                elapsed = time.perf_counter() - epoch_start_time
                pct = 100.0 * n_batches / total_batches
                rate = n_batches / max(elapsed, 0.001)
                eta = (total_batches - n_batches) / max(rate, 0.001)
                avg_loss = epoch_loss / n_batches
                print(f"[EXIT TRAIN] Epoch {epoch+1} batch {n_batches}/{total_batches} ({pct:.0f}%) loss={avg_loss:.4f} ETA={eta:.0f}s", flush=True)
            
            # TensorBoard step logging
            if global_step % TB_STEP_LOG_EVERY == 0:
                writer.add_scalar("exit/loss_train_step", loss.item(), global_step)
                writer.add_scalar("exit/grad_norm", grad_norm, global_step)
                
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    batch_exit_rate = preds.mean().item()
                    writer.add_scalar("exit/batch_exit_rate", batch_exit_rate, global_step)
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            val_loss = criterion(val_logits, val_y).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            if clog and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(device) / 1e9
                reserved = torch.cuda.memory_reserved(device) / 1e9
                clog.info(
                    ComponentLogger.PERF,
                    "Exit validation GPU mem allocated={:.2f}GB reserved={:.2f}GB".format(allocated, reserved),
                )
            val_preds = (val_probs >= 0.5).astype(int)
            val_true = val_y.cpu().numpy().astype(int)
            
            val_acc = np.mean(val_preds == val_true)
            val_exit_rate = np.mean(val_preds == EXIT_EXIT)
            val_true_exit_rate = np.mean(val_true == EXIT_EXIT)
            
            # Precision/recall for EXIT class
            tp = np.sum((val_preds == EXIT_EXIT) & (val_true == EXIT_EXIT))
            fp = np.sum((val_preds == EXIT_EXIT) & (val_true == EXIT_HOLD))
            fn = np.sum((val_preds == EXIT_HOLD) & (val_true == EXIT_EXIT))
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            
            # ================================================================
            # PnL Proxy: Simulate trades using exit policy on validation data
            # ================================================================
            # metadata columns: [entry_idx, bar_idx, side]
            # Features contain: unrealized_pnl_atr at index -3, mae at -2, mfe at -1
            val_metadata_np = val_metadata.cpu().numpy() if hasattr(val_metadata, 'cpu') else val_metadata
            val_features_np = val_X.cpu().numpy()
            
            # Group by entry (trade) - find first EXIT prediction for each trade
            pnl_scores = []
            hold_times = []
            
            # Get unique entries in validation set
            unique_entries = np.unique(val_metadata_np[:, 0])
            if val_entries_limit is not None and val_entries_limit > 0 and val_entries_limit < len(unique_entries):
                rng = np.random.default_rng(42)
                sampled = rng.choice(unique_entries, size=val_entries_limit, replace=False)
                unique_entries = np.sort(sampled)
            if clog:
                limit_info = f" (limited to {len(unique_entries)} entries)" if val_entries_limit else ""
                clog.info(
                    ComponentLogger.TRAIN,
                    f"Simulating exit PnL proxy for {len(unique_entries)} unique entries (validation){limit_info}",
                )
            
            for entry_idx in unique_entries:
                # Get all samples for this entry
                entry_mask = val_metadata_np[:, 0] == entry_idx
                entry_probs = val_probs[entry_mask]
                entry_features = val_features_np[entry_mask]
                entry_meta = val_metadata_np[entry_mask]
                
                if len(entry_probs) == 0:
                    continue
                
                # Find first bar where p_exit >= threshold (or use max_hold)
                exit_threshold_val = 0.5  # Use 0.5 for validation (training threshold)
                exit_mask = entry_probs >= exit_threshold_val
                
                if np.any(exit_mask):
                    first_exit_idx = np.argmax(exit_mask)  # First True
                else:
                    first_exit_idx = len(entry_probs) - 1  # Max hold - exit at end
                
                # Get PnL at exit point (unrealized_pnl_atr is at index -3)
                exit_pnl_atr = entry_features[first_exit_idx, -3]
                hold_time = first_exit_idx + 1  # Bars held
                
                pnl_scores.append(exit_pnl_atr)
                hold_times.append(hold_time)
            
            if len(pnl_scores) > 0:
                pnl_scores = np.array(pnl_scores)
                hold_times = np.array(hold_times)
                
                avg_pnl = float(np.mean(pnl_scores))
                total_pnl = float(np.sum(pnl_scores))
                win_rate = float(np.mean(pnl_scores > 0))
                avg_hold_time = float(np.mean(hold_times))
                avg_pnl_winners = float(np.mean(pnl_scores[pnl_scores > 0])) if np.any(pnl_scores > 0) else 0.0
                avg_pnl_losers = float(np.mean(pnl_scores[pnl_scores <= 0])) if np.any(pnl_scores <= 0) else 0.0
            else:
                avg_pnl = total_pnl = win_rate = avg_hold_time = 0.0
                avg_pnl_winners = avg_pnl_losers = 0.0
        
        # Log epoch metrics
        writer.add_scalar("exit/loss_train", avg_train_loss, epoch)
        writer.add_scalar("exit/loss_val", val_loss, epoch)
        writer.add_scalar("exit/val_accuracy", val_acc, epoch)
        writer.add_scalar("exit/val_exit_rate_pred", val_exit_rate, epoch)
        writer.add_scalar("exit/val_exit_rate_true", val_true_exit_rate, epoch)
        writer.add_scalar("exit/val_precision", precision, epoch)
        writer.add_scalar("exit/val_recall", recall, epoch)
        writer.add_scalar("exit/val_f1", f1, epoch)
        
        # PnL proxy metrics
        writer.add_scalar("exit_pnl/avg_pnl_atr", avg_pnl, epoch)
        writer.add_scalar("exit_pnl/total_pnl_atr", total_pnl, epoch)
        writer.add_scalar("exit_pnl/win_rate", win_rate, epoch)
        writer.add_scalar("exit_pnl/avg_hold_bars", avg_hold_time, epoch)
        writer.add_scalar("exit_pnl/avg_winner_atr", avg_pnl_winners, epoch)
        writer.add_scalar("exit_pnl/avg_loser_atr", avg_pnl_losers, epoch)
        
        if clog:
            clog.info(
                ComponentLogger.TRAIN,
                f"Epoch {epoch+1}/{EXIT_EPOCHS}: loss={avg_train_loss:.4f}/{val_loss:.4f} "
                f"f1={f1:.3f} pnl={avg_pnl:.2f} win={win_rate:.1%} hold={avg_hold_time:.1f}bars"
            )
        
        # Use F1 as selection metric
        val_metric = f1
        
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch
            patience_counter = 0
            
            # Save best checkpoint
            torch.save({
                "model_state_dict": model.state_dict(),
                "in_dim": in_dim,
                "hidden": EXIT_HIDDEN,
                "dropout": EXIT_DROPOUT,
                "max_hold_bars": max_hold_bars,
                "lambda_dd": lambda_dd,
                "label_margin": label_margin,
                "score_min": score_min,
                "epoch": epoch,
                "val_f1": f1,
                "val_acc": val_acc,
            }, best_ckpt_path)
            
            if clog:
                clog.info(ComponentLogger.IO, f"Saved best exit checkpoint: epoch {epoch+1} val_f1={f1:.4f}")
            
            writer.add_scalar("exit/is_best", 1, epoch)
        else:
            patience_counter += 1
            writer.add_scalar("exit/is_best", 0, epoch)
        
        if patience_counter >= EXIT_EARLY_STOP_PATIENCE:
            if clog:
                clog.info(ComponentLogger.TRAIN, f"Early stopping at epoch {epoch+1}")
            break
    
    writer.close()
    
    if clog:
        clog.info(ComponentLogger.BOOT, "=" * 70)
        clog.info(ComponentLogger.BOOT, f"EXIT POLICY TRAINING COMPLETE")
        clog.info(ComponentLogger.BOOT, f"Best epoch: {best_epoch+1}, Best F1: {best_val_metric:.4f}")
        clog.info(ComponentLogger.BOOT, f"Checkpoint: {best_ckpt_path}")
        clog.info(ComponentLogger.BOOT, "=" * 70)
    
    return best_ckpt_path


def run_live(
    adapters: list[ModelAdapter],
    meta_ckpt: Path,
    exit_ckpt: Optional[Path] = None,
    dry_run: bool = False,
    heartbeat: bool = True,
    log_every_seconds: float = 1.0,
    exit_prob_threshold: float = EXIT_PROB_THRESHOLD,
    exit_max_hold_bars: int = EXIT_MAX_HOLD_BARS,
    rebuild_features_every_tick: bool = False,
) -> None:
    """
    Run live trading with meta-policy ensemble and learned exit policy.
    
    Entry policy (meta-policy): BUY / SELL / NO_TRADE decisions when flat
    Exit policy (ExitPolicyNet): HOLD / EXIT decisions when in a trade
    
    Args:
        adapters: List of base model adapters
        meta_ckpt: Path to meta-policy checkpoint
        exit_ckpt: Path to exit policy checkpoint (optional, will look for default)
        dry_run: If True, log but don't send orders
        heartbeat: If True, log status every second
        log_every_seconds: Seconds between heartbeat logs
        exit_prob_threshold: p_exit threshold to trigger exit
        exit_max_hold_bars: Force exit after this many bars
    """
    if clog:
        clog.info(ComponentLogger.BOOT, f"Starting live runner dry_run={dry_run} heartbeat={heartbeat}")
        clog.info(ComponentLogger.BOOT, f"Exit policy: threshold={exit_prob_threshold} max_hold={exit_max_hold_bars}")
    
    if mt5 is None:
        if clog:
            clog.error(ComponentLogger.ERROR, "MetaTrader5 module unavailable")
        raise RuntimeError("MetaTrader5 module unavailable.")
    
    if not _mt5_initialize():
        err = mt5.last_error() if mt5 else "unknown"
        if clog:
            clog.error(ComponentLogger.ERROR, f"MT5 init failed: {err}")
        raise RuntimeError(f"MT5 init failed: {err}")
    
    acct = mt5.account_info()
    login = getattr(acct, "login", None) if acct else None
    balance = getattr(acct, "balance", 0) if acct else 0
    if clog:
        clog.info(ComponentLogger.BOOT, f"MT5 connected login={login} balance={balance}")
    mt5.symbol_select(SYMBOL, True)
    
    # Load meta-policy (entry)
    if clog:
        clog.info(ComponentLogger.IO, f"Loading meta-policy checkpoint path={meta_ckpt}")
    ckpt = torch.load(meta_ckpt, map_location="cpu")
    in_dim = ckpt["in_dim"]
    meta_policy = MetaPolicyNet(in_dim)
    meta_policy.load_state_dict(ckpt["state_dict"])
    meta_policy.eval()
    if clog:
        clog.info(ComponentLogger.META, f"Meta-policy (entry) loaded in_dim={in_dim} eval_mode=True")
    
    # Load exit policy
    exit_policy = None
    if exit_ckpt is None:
        # Try to find default exit checkpoint
        default_exit_ckpt = EXIT_CHECKPOINT_DIR / "exit_policy_best.pt"
        if default_exit_ckpt.exists():
            exit_ckpt = default_exit_ckpt
    
    if exit_ckpt is not None and exit_ckpt.exists():
        if clog:
            clog.info(ComponentLogger.IO, f"Loading exit policy checkpoint path={exit_ckpt}")
        try:
            exit_ckpt_data = torch.load(exit_ckpt, map_location="cpu")
        except Exception as exc:
            # PyTorch 2.6+ defaults to weights_only=True; allow safe globals and retry with weights_only=False
            try:
                import numpy as _np  # Local import to keep scope tight
                torch.serialization.add_safe_globals([_np.core.multiarray.scalar])
            except Exception:
                pass
            if clog:
                clog.warning(ComponentLogger.IO, f"Default load failed ({exc}); retrying with weights_only=False")
            exit_ckpt_data = torch.load(exit_ckpt, map_location="cpu", weights_only=False)
        exit_in_dim = exit_ckpt_data["in_dim"]
        exit_policy = ExitPolicyNet(
            in_dim=exit_in_dim,
            hidden=exit_ckpt_data.get("hidden", EXIT_HIDDEN),
            dropout=exit_ckpt_data.get("dropout", EXIT_DROPOUT),
        )
        exit_policy.load_state_dict(exit_ckpt_data["model_state_dict"])
        exit_policy.eval()
        if clog:
            clog.info(ComponentLogger.META, f"Exit policy loaded in_dim={exit_in_dim} eval_mode=True")
    else:
        if clog:
            clog.warning(ComponentLogger.BOOT, "No exit policy checkpoint found - using max_hold_bars only for exits")
    
    # TensorBoard
    tb_dir = OUTPUT_DIR / "runs" / f"meta_live_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=tb_dir)
    if clog:
        clog.info(ComponentLogger.IO, f"TensorBoard logs at {tb_dir}")
    
    # Live log file
    live_log_path = LOGS_DIR / LIVE_LOG_FILENAME
    if not live_log_path.exists():
        with live_log_path.open("w", newline="", encoding="utf-8") as fh:
            header = "bar_ts,action,action_name,p_buy,p_sell,p_no_trade,p_trend,vol_30m,spread"
            for a in adapters:
                header += f",{a.name}_p_trade,{a.name}_p_dir"
            header += ",lot,total_lots\n"
            fh.write(header)
    
    bar_count = 0
    action_buf = deque(maxlen=1000)
    last_heartbeat = time.time()
    
    # Caching for feature frames (exactly like base models do)
    cached_dfs: list[pd.DataFrame | None] = [None] * len(adapters)
    cached_feature_cols: list[list[str] | None] = [None] * len(adapters)
    last_closed_bar_ts: Any = None
    
    # Trade state tracking for exit policy
    trade_state: dict[str, Any] = {
        "active": False,
        "side": 0,  # +1 for long, -1 for short
        "entry_price": 0.0,
        "entry_atr": 0.0,
        "entry_bar": 0,
        "entry_time": None,
        "ticket": 0,
        "volume": 0.0,
        "prices_since_entry": [],  # List of (high, low) tuples
    }
    
    # Warmup: Test that feature building works before entering main loop
    if clog:
        clog.info(ComponentLogger.BOOT, "Testing feature building functions...")
    print("[WARMUP] Testing module functions...", flush=True)
    for i, adapter in enumerate(adapters):
        print(f"[WARMUP] Testing {adapter.name}...", flush=True)
        # Test that the function exists and is accessible
        func = getattr(adapter.module, '_build_feature_frame_from_sources', None)
        print(f"[WARMUP] {adapter.name} function found: {func is not None}", flush=True)
        # Check globals that might affect behavior
        use_levels = getattr(adapter.module, 'USE_LEVELS_KMEANS', 'N/A')
        use_channel = getattr(adapter.module, 'USE_CHANNEL_QUANTREG', 'N/A')
        use_regime = getattr(adapter.module, 'USE_BOOK_REGIME', False)
        print(f"[WARMUP] {adapter.name} USE_LEVELS={use_levels} USE_CHANNEL={use_channel} USE_REGIME={use_regime}", flush=True)
    print("[WARMUP] All modules checked", flush=True)
    
    if clog:
        clog.info(ComponentLogger.BOOT, "Entering live loop...")
    
    while True:
        loop_start = time.perf_counter()
        try:
            # Fetch latest data (use first adapter's module for data fetching)
            with timed_block(ComponentLogger.DATA, "fetch_bars") as fetch_timing:
                mod = adapters[0].module
                xau = mod._fetch_m1(SYMBOL, LIVE_FETCH_BARS)
                xag = mod._fetch_m1(getattr(mod, "XAG_SYMBOL", "XAGUSD"), LIVE_FETCH_BARS)
                dxy = mod._fetch_m1(getattr(mod, "USD_INDEX_SYMBOL", "DXY"), LIVE_FETCH_BARS)
            
            if xau is None or len(xau) < 100:
                if clog:
                    clog.debug(ComponentLogger.DATA, f"Waiting for data: XAU rows={len(xau) if xau is not None else 0}")
                time.sleep(1)
                continue
            if xag is None or dxy is None:
                if clog:
                    clog.warning(ComponentLogger.DATA, f"Missing data: XAG={xag is not None} DXY={dxy is not None}")
                time.sleep(1)
                continue
            
            tick = _get_tick(SYMBOL)
            if tick is None:
                if clog:
                    clog.debug(ComponentLogger.DATA, "No tick available")
                time.sleep(1)
                continue
            
            # Detect new closed bar (exactly like base models)
            closed_bar_ts = xau['bar_dt'].iloc[-2] if len(xau) >= 2 else None
            bar_dt_latest = xau['bar_dt'].iloc[-1] if 'bar_dt' in xau.columns else "unknown"
            new_closed_bar = last_closed_bar_ts is None or (closed_bar_ts is not None and closed_bar_ts != last_closed_bar_ts)
            
            if clog:
                clog.info(ComponentLogger.DATA, f"Data: XAU={len(xau)} XAG={len(xag)} DXY={len(dxy)} bars, new_closed_bar={new_closed_bar}")
            
            # Only infer on closed bars (once per minute on M1 data)
            if not new_closed_bar:
                time.sleep(LIVE_POLL_SECONDS)
                continue
            
            # Build features - simplified direct call
            dfs = []
            for i, adapter in enumerate(adapters):
                if cached_dfs[i] is None or new_closed_bar or rebuild_features_every_tick:
                    print(f"[{adapter.name}] Building features from {len(xau)} bars...", flush=True)
                    feat_start = time.perf_counter()
                    
                    # Direct function call - no wrappers
                    try:
                        df, feature_cols = adapter.module._build_feature_frame_from_sources(
                            xau.copy(), xag.copy(), dxy.copy(), log_progress=True, live_fill=True
                        )
                        print(f"[{adapter.name}] Base features done", flush=True)
                    except Exception as e:
                        print(f"[{adapter.name}] ERROR in _build_feature_frame_from_sources: {e}", flush=True)
                        raise
                    
                    # Add regime features if needed
                    if getattr(adapter.module, 'USE_BOOK_REGIME', False):
                        print(f"[{adapter.name}] Adding regime features...", flush=True)
                        try:
                            params = adapter.module._ensure_regime_params(df, use_mt5=True)
                            df, feature_cols = adapter.module._add_regime_features(df, feature_cols, params)
                            print(f"[{adapter.name}] Regime features done", flush=True)
                        except Exception as e:
                            print(f"[{adapter.name}] ERROR in regime features: {e}", flush=True)
                            raise
                    
                    cached_dfs[i] = df
                    cached_feature_cols[i] = feature_cols
                    feat_ms = (time.perf_counter() - feat_start) * 1000
                    print(f"[{adapter.name}] COMPLETE: {feat_ms:.0f}ms, {len(df)} rows", flush=True)
                else:
                    # Fast update for same bar
                    if hasattr(adapter.module, '_fast_update_live_features'):
                        cached_dfs[i] = adapter.module._fast_update_live_features(cached_dfs[i], xau, xag, dxy)
                dfs.append(cached_dfs[i])
            
            if new_closed_bar:
                last_closed_bar_ts = closed_bar_ts
            
            # Get predictions from each model
            with timed_block(ComponentLogger.INFER, "model_inference") as infer_timing:
                model_outputs = []
                for i, (adapter, df) in enumerate(zip(adapters, dfs, strict=True)):
                    if clog:
                        clog.debug(ComponentLogger.INFER, f"Running inference for model {i+1}/{len(adapters)}: {adapter.name}")
                    infer_start = time.perf_counter()
                    idx = len(df) - 1
                    norm_matrix = adapter._norm_matrix(df)
                    short, mid, long, slow = adapter._build_seq_at(norm_matrix, df, idx)
                    short_t = torch.from_numpy(short).unsqueeze(0).to(adapter.device)
                    mid_t = torch.from_numpy(mid).unsqueeze(0).to(adapter.device)
                    long_t = torch.from_numpy(long).unsqueeze(0).to(adapter.device)
                    slow_t = torch.from_numpy(slow).unsqueeze(0).to(adapter.device)
                    
                    p_trade, p_dir = adapter._infer_single(short_t, mid_t, long_t, slow_t)
                    model_outputs.append((float(p_trade[0]), float(p_dir[0])))
                    infer_ms = (time.perf_counter() - infer_start) * 1000
                    if clog:
                        clog.info(ComponentLogger.INFER, f"Model {adapter.name}: p_trade={p_trade[0]:.3f} p_dir={p_dir[0]:.3f} ({infer_ms:.1f}ms)")
            
            # Build meta-policy features
            base_df = dfs[0]
            close_arr = base_df["xau_close"].to_numpy(dtype=np.float64)
            vol_arr = base_df["vol_30m"].to_numpy(dtype=np.float64)
            atr_px = vol_arr * close_arr
            p_trend = _compute_p_trend_from_series(close_arr, atr_px, len(base_df) - 1)
            
            # Extract context variables for use in trade state and exit policy
            vol_30m = float(base_df["vol_30m"].iloc[-1]) if "vol_30m" in base_df.columns else 0.0001
            spread = float(base_df["xau_spread"].iloc[-1]) if "xau_spread" in base_df.columns else 0.0
            close_price = float(base_df["xau_close"].iloc[-1])
            
            # Build features (must match training feature order!)
            features = []
            
            # 1. Model outputs
            for p_trade, p_dir in model_outputs:
                features.extend([p_trade, p_dir])
            
            # 2. Engineered trend
            features.append(p_trend)
            
            # 3. Basic context
            features.append(vol_30m)
            features.append(spread)
            
            # 4. Time context
            for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
                if col in base_df.columns:
                    features.append(float(base_df[col].iloc[-1]))
            
            # 5. Correlation asset context
            for col in ["xag_ret_1", "dxy_ret_1", "gold_silver_ratio"]:
                if col in base_df.columns:
                    val = base_df[col].iloc[-1]
                    features.append(0.0 if pd.isna(val) else float(val))
            
            # 6. Volatility/momentum context
            for col in ["atr_pct", "rsi_14", "ret_15"]:
                if col in base_df.columns:
                    val = base_df[col].iloc[-1]
                    features.append(0.0 if pd.isna(val) else float(val))
            
            feats_t = torch.tensor([features], dtype=torch.float32)
            
            # Get meta-policy prediction
            with timed_block(ComponentLogger.META, "meta_inference") as meta_timing:
                if bar_count < LIVE_WARMUP_BARS:
                    # During warmup, default to NO_TRADE
                    action = ACTION_NO_TRADE
                    probs = np.array([0.33, 0.33, 0.34])
                    reason = "warmup"
                else:
                    with torch.no_grad():
                        logits = meta_policy(feats_t)
                        probs = F.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
                    action = int(np.argmax(probs))
                    reason = "meta_decision"
            
            action_name = ACTION_NAMES[action]
            confidence = float(probs[action])
            
            # Get current positions
            positions = mt5.positions_get(symbol=SYMBOL) or []
            symbol_positions = [pos for pos in positions if getattr(pos, "symbol", "") == SYMBOL]
            total_lots = sum(pos.volume for pos in symbol_positions)
            
            # Determine current position direction (None if flat, "BUY" or "SELL" if open)
            current_direction = None
            if symbol_positions:
                pos_type = getattr(symbol_positions[0], "type", None)
                if pos_type == mt5.POSITION_TYPE_BUY:
                    current_direction = "BUY"
                elif pos_type == mt5.POSITION_TYPE_SELL:
                    current_direction = "SELL"
            
            # Sync internal trade state with MT5
            if current_direction is None:
                # We're flat - reset trade state
                if trade_state["active"]:
                    if clog:
                        clog.info(ComponentLogger.META, "Position closed externally, resetting trade state")
                trade_state["active"] = False
                trade_state["prices_since_entry"] = []
            elif not trade_state["active"]:
                # Position exists but we don't have state - sync from MT5
                pos = symbol_positions[0]
                trade_state["active"] = True
                trade_state["side"] = 1 if current_direction == "BUY" else -1
                trade_state["entry_price"] = float(getattr(pos, "price_open", tick.last))
                trade_state["entry_atr"] = float(vol_30m * tick.last) if vol_30m > 0 else 1.0
                trade_state["entry_bar"] = bar_count
                trade_state["entry_time"] = datetime.datetime.utcnow()
                trade_state["ticket"] = int(getattr(pos, "ticket", 0))
                trade_state["volume"] = float(getattr(pos, "volume", 0))
                trade_state["prices_since_entry"] = []
                if clog:
                    clog.info(ComponentLogger.META, f"Synced trade state from MT5: side={current_direction} entry_price={trade_state['entry_price']:.2f}")
            
            # Confidence gate for entries: if flat and low confidence, force NO_TRADE
            if not trade_state["active"] and action != ACTION_NO_TRADE and confidence < LIVE_MIN_CONF:
                action = ACTION_NO_TRADE
                action_name = ACTION_NAMES[action]
                reason = f"conf_gate<{LIVE_MIN_CONF}"
            
            # Update prices_since_entry if we have a position
            if trade_state["active"]:
                # Use current high/low from latest bar
                current_high = float(tick.ask) if tick else close_price
                current_low = float(tick.bid) if tick else close_price
                trade_state["prices_since_entry"].append((current_high, current_low))
            
            # =====================================================================
            # LEARNED EXIT POLICY (replaces flip-close logic)
            # =====================================================================
            p_exit = 0.0
            exit_decision = "HOLD"
            should_exit = False
            close_reason = ""
            
            if trade_state["active"]:
                # Calculate trade state features
                age_bars = bar_count - trade_state["entry_bar"]
                current_price = float(tick.last) if tick and tick.last > 0 else float((tick.bid + tick.ask) / 2)
                
                # Compute trade metrics
                prices_arr = np.array(trade_state["prices_since_entry"]) if trade_state["prices_since_entry"] else np.array([[current_price, current_price]])
                unrealized_pnl_atr, mae_atr, mfe_atr, norm_age = _compute_exit_trade_state(
                    trade_state["entry_price"],
                    trade_state["entry_atr"],
                    current_price,
                    trade_state["side"],
                    prices_arr,
                    age_bars,
                    exit_max_hold_bars,
                )
                
                # Check exit conditions
                # 1. Max hold time reached
                if age_bars >= exit_max_hold_bars:
                    should_exit = True
                    close_reason = "max_hold_time"
                    exit_decision = "EXIT"
                # 2. Exit policy decision (if we have the model)
                elif exit_policy is not None:
                    # Build exit feature vector
                    exit_feat = []
                    
                    # Base model outputs
                    for p_trade_val, p_dir_val in model_outputs:
                        exit_feat.append(float(p_trade_val))
                        exit_feat.append(float(p_dir_val))
                    
                    # Context features
                    exit_feat.append(float(p_trend))
                    exit_feat.append(float(vol_30m))
                    exit_feat.append(float(spread))
                    
                    # Trade state features
                    exit_feat.append(float(trade_state["side"]))
                    exit_feat.append(float(norm_age))
                    exit_feat.append(float(unrealized_pnl_atr))
                    exit_feat.append(float(mae_atr))
                    exit_feat.append(float(mfe_atr))
                    
                    # Run exit policy inference
                    exit_feat_t = torch.tensor([exit_feat], dtype=torch.float32)
                    with torch.no_grad():
                        exit_logit = exit_policy(exit_feat_t)
                        p_exit = float(torch.sigmoid(exit_logit).item())
                    
                    if p_exit >= exit_prob_threshold:
                        should_exit = True
                        close_reason = f"exit_policy_p{p_exit:.2f}"
                        exit_decision = "EXIT"
                    else:
                        exit_decision = "HOLD"
                
                # Execute exit if needed
                if should_exit:
                    tick_now = _get_tick(SYMBOL)
                    for pos in symbol_positions:
                        pos_volume = float(pos.volume)
                        pos_ticket = int(pos.ticket)
                        pos_type = getattr(pos, "type", None)
                        
                        if pos_type == mt5.POSITION_TYPE_BUY:
                            close_price = tick_now.bid
                            close_type = mt5.ORDER_TYPE_SELL
                        else:
                            close_price = tick_now.ask
                            close_type = mt5.ORDER_TYPE_BUY
                        
                        close_request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": SYMBOL,
                            "volume": pos_volume,
                            "type": close_type,
                            "position": pos_ticket,
                            "price": close_price,
                            "deviation": 20,
                            "magic": 99002,
                            "comment": f"META_EXIT_{close_reason}",
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        
                        if clog:
                            clog.info(
                                ComponentLogger.EXEC,
                                f"EXIT TRADE: ticket={pos_ticket} volume={pos_volume:.2f} "
                                f"reason={close_reason} p_exit={p_exit:.2f} age={age_bars} pnl_atr={unrealized_pnl_atr:.2f}"
                            )
                        
                        if dry_run:
                            if clog:
                                clog.info(ComponentLogger.EXEC, f"DRY_RUN: Would exit position ticket={pos_ticket}")
                        else:
                            close_res = mt5.order_send(close_request)
                            close_retcode = close_res.retcode if close_res else -1
                            close_retcode_name = MT5_RETCODE_MAP.get(close_retcode, "UNKNOWN")
                            if clog:
                                clog.info(ComponentLogger.EXEC, f"Exit result: retcode={close_retcode}({close_retcode_name})")
                            if close_retcode != 10009:
                                if clog:
                                    clog.warning(ComponentLogger.EXEC, f"Exit may have failed: {mt5.last_error()}")
                            else:
                                # Reset trade state after successful close
                                trade_state["active"] = False
                                trade_state["prices_since_entry"] = []
                    
                    # Update position info after closing
                    total_lots = 0.0
                    current_direction = None
            
            # =====================================================================
            # ENTRY POLICY (only when flat)
            # =====================================================================
            can_trade = False
            if current_direction is None:  # We are flat
                can_trade = action != ACTION_NO_TRADE and total_lots < LIVE_MAX_LOTS
                if action == ACTION_NO_TRADE:
                    reason = "meta_no_trade"
                elif total_lots >= LIVE_MAX_LOTS:
                    reason = "max_lots_reached"
                    can_trade = False
                else:
                    reason = "entry_signal"
            else:
                # Position is open - entry policy doesn't apply
                reason = f"in_trade_{exit_decision}"
            
            # Execute entry trade if action is BUY or SELL and we're flat
            lot = 0.0
            exec_result = None
            if can_trade:
                lot = min(LIVE_MAX_LOTS - total_lots, 1.0)
                lot = _round_lot(max(LIVE_LOT_STEP, lot), LIVE_LOT_STEP)
                tick_now = _get_tick(SYMBOL)
                
                if action == ACTION_BUY:
                    price = tick_now.ask
                    order_type = mt5.ORDER_TYPE_BUY
                    side = "BUY"
                    side_int = 1
                else:  # SELL
                    price = tick_now.bid
                    order_type = mt5.ORDER_TYPE_SELL
                    side = "SELL"
                    side_int = -1
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "volume": lot,
                    "type": order_type,
                    "price": price,
                    "deviation": 20,
                    "magic": 99002,
                    "comment": f"META_{action_name}",
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                if clog:
                    clog.info(
                        ComponentLogger.EXEC,
                        f"ENTRY: action={side} symbol={SYMBOL} lots={lot:.2f} price={price:.2f}"
                    )
                    clog.debug(ComponentLogger.EXEC, f"Request dict: {json.dumps(request, default=str)}")
                
                if dry_run:
                    if clog:
                        clog.info(ComponentLogger.EXEC, f"DRY_RUN: Would send entry order action={side} lots={lot:.2f}")
                    exec_result = "dry_run"
                    # Update trade state for dry run
                    trade_state["active"] = True
                    trade_state["side"] = side_int
                    trade_state["entry_price"] = price
                    trade_state["entry_atr"] = float(vol_30m * price) if vol_30m > 0 else 1.0
                    trade_state["entry_bar"] = bar_count
                    trade_state["entry_time"] = datetime.datetime.utcnow()
                    trade_state["volume"] = lot
                    trade_state["prices_since_entry"] = []
                else:
                    with timed_block(ComponentLogger.EXEC, "order_send") as exec_timing:
                        res = mt5.order_send(request)
                    
                    retcode = res.retcode if res else -1
                    retcode_name = MT5_RETCODE_MAP.get(retcode, "UNKNOWN")
                    ticket = res.order if res else None
                    
                    if clog:
                        clog.info(
                            ComponentLogger.EXEC,
                            f"Entry result: retcode={retcode}({retcode_name}) ticket={ticket} "
                            f"elapsed_ms={exec_timing['elapsed_ms']:.2f}"
                        )
                    
                    if retcode == 10009:  # Success
                        # Update trade state
                        trade_state["active"] = True
                        trade_state["side"] = side_int
                        trade_state["entry_price"] = price
                        trade_state["entry_atr"] = float(vol_30m * price) if vol_30m > 0 else 1.0
                        trade_state["entry_bar"] = bar_count
                        trade_state["entry_time"] = datetime.datetime.utcnow()
                        trade_state["ticket"] = ticket if ticket else 0
                        trade_state["volume"] = lot
                        trade_state["prices_since_entry"] = []
                    else:
                        if clog:
                            last_err = mt5.last_error()
                            clog.warning(ComponentLogger.EXEC, f"Entry may have failed: last_error={last_err}")
                    
                    exec_result = f"{retcode}:{retcode_name}"
            
            # Logging
            action_buf.append(action)
            
            # Heartbeat logging (every second)
            now = time.time()
            if heartbeat and (now - last_heartbeat) >= log_every_seconds:
                # Build compact heartbeat line
                model_str = " ".join([
                    f"{adapters[i].name[:3]}(pT={mo[0]:.2f},pD={mo[1]:.2f})"
                    for i, mo in enumerate(model_outputs)
                ])
                meta_str = f"meta[B={probs[ACTION_BUY]:.2f},S={probs[ACTION_SELL]:.2f},N={probs[ACTION_NO_TRADE]:.2f}]"
                
                # Add exit policy info when in a trade
                if trade_state["active"]:
                    age_bars = bar_count - trade_state["entry_bar"]
                    side_str = "LONG" if trade_state["side"] > 0 else "SHORT"
                    # Compute current PnL
                    current_price = float(tick.last) if tick and tick.last > 0 else float((tick.bid + tick.ask) / 2)
                    entry_atr_safe = max(trade_state.get("entry_atr", 0.0), 1e-8)
                    if trade_state["side"] > 0:
                        pnl_atr = (current_price - trade_state["entry_price"]) / entry_atr_safe
                    else:
                        pnl_atr = (trade_state["entry_price"] - current_price) / entry_atr_safe
                    exit_str = f"trade[{side_str} age={age_bars} pnl={pnl_atr:.2f} p_exit={p_exit:.2f} {exit_decision}]"
                else:
                    exit_str = "trade[FLAT]"
                
                if clog:
                    clog.info(
                        ComponentLogger.HEARTBEAT,
                        f"bar={bar_count} ts={bar_dt_latest} {model_str} {meta_str} "
                        f"{exit_str} action={action_name} conf={confidence:.2f} lots={total_lots:.2f} reason=\"{reason}\""
                    )
                last_heartbeat = now
            
            # TensorBoard logging
            if LIVE_LOG_EVERY_BARS > 0 and (bar_count % LIVE_LOG_EVERY_BARS) == 0:
                writer.add_scalar("live/action", action, bar_count)
                writer.add_scalar("live/p_buy", probs[ACTION_BUY], bar_count)
                writer.add_scalar("live/p_sell", probs[ACTION_SELL], bar_count)
                writer.add_scalar("live/p_no_trade", probs[ACTION_NO_TRADE], bar_count)
                writer.add_scalar("live/p_trend", p_trend, bar_count)
                writer.add_scalar("live/total_lots", total_lots, bar_count)
                
                # Trade rate
                if action_buf:
                    trade_rate = 1.0 - np.mean(np.array(list(action_buf)) == ACTION_NO_TRADE)
                    writer.add_scalar("live/trade_rate", trade_rate, bar_count)
                
                for i, (a, (p_trade, p_dir)) in enumerate(zip(adapters, model_outputs)):
                    writer.add_scalar(f"live/{a.name}_p_trade", p_trade, bar_count)
                    writer.add_scalar(f"live/{a.name}_p_dir", p_dir, bar_count)
                
                writer.flush()
            
            # Write to log file
            with live_log_path.open("a", newline="", encoding="utf-8") as fh:
                line = f"{bar_dt_latest},{action},{action_name},{probs[0]:.6f},{probs[1]:.6f},{probs[2]:.6f},{p_trend:.6f},{features[-2]:.6f},{features[-1]:.6f}"
                for p_trade, p_dir in model_outputs:
                    line += f",{p_trade:.6f},{p_dir:.6f}"
                line += f",{lot:.2f},{total_lots:.2f}\n"
                fh.write(line)
            
            bar_count += 1
            
            # Sleep to maintain ~1 second loop
            loop_elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, LIVE_POLL_SECONDS - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        except Exception as exc:
            if clog:
                clog.exception(ComponentLogger.ERROR, f"Live loop error: {exc}")
            else:
                logging.exception("Meta-policy live error: %s", exc)
            # Check if this is a fatal error (NaNs in features)
            if "NaN" in str(exc) or "Inf" in str(exc):
                if clog:
                    clog.error(ComponentLogger.ERROR, "Fatal: NaN/Inf detected, stopping live loop")
                raise
            time.sleep(5)


# =============================================================================
# CLI and main
# =============================================================================
def parse_model_specs(modules_str: str, ckpts_str: str) -> list[dict[str, Any]]:
    """Parse CLI model specifications."""
    modules = [m.strip() for m in modules_str.split(",")]
    ckpts = [c.strip() for c in ckpts_str.split(",")]
    
    if len(modules) != len(ckpts):
        raise ValueError(f"Number of modules ({len(modules)}) != number of checkpoints ({len(ckpts)})")
    
    specs = []
    for i, (mod_path, ckpt_path) in enumerate(zip(modules, ckpts)):
        # Determine kind from filename
        mod_name = Path(mod_path).stem.lower()
        if "regime" in mod_name:
            kind = "regime"
        elif "15" in mod_name or "mom" in mod_name:
            kind = "mom15"
        else:
            kind = "mr"
        
        specs.append({
            "name": f"Model{i+1}_{kind}",
            "script_path": mod_path,
            "ckpt_path": ckpt_path,
            "kind": kind,
        })
    
    return specs


def log_boot_info(args: argparse.Namespace, specs: list[dict[str, Any]]) -> None:
    """Log comprehensive boot information."""
    if not clog:
        return
    
    clog.info(ComponentLogger.BOOT, "=" * 70)
    clog.info(ComponentLogger.BOOT, "Meta-Policy Ensemble Starting")
    clog.info(ComponentLogger.BOOT, "=" * 70)
    
    # Environment
    clog.info(ComponentLogger.BOOT, f"Python: {sys.version}")
    clog.info(ComponentLogger.BOOT, f"PyTorch: {torch.__version__}")
    clog.info(ComponentLogger.BOOT, f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        clog.info(ComponentLogger.BOOT, f"CUDA device: {torch.cuda.get_device_name(0)}")
    clog.info(ComponentLogger.BOOT, f"NumPy: {np.__version__}")
    clog.info(ComponentLogger.BOOT, f"Pandas: {pd.__version__}")
    
    # Paths
    clog.info(ComponentLogger.IO, f"Base dir: {BASE_DIR}")
    clog.info(ComponentLogger.IO, f"Output dir: {OUTPUT_DIR}")
    clog.info(ComponentLogger.IO, f"Checkpoint dir: {CHECKPOINT_DIR}")
    clog.info(ComponentLogger.IO, f"Logs dir: {LOGS_DIR}")
    clog.info(ComponentLogger.IO, f"TensorBoard dir: {OUTPUT_DIR / 'tb'}")
    
    # Mode
    mode = "LIVE" if args.live else "TRAIN"
    clog.info(ComponentLogger.BOOT, f"Mode: {mode}")
    if hasattr(args, 'dry_run'):
        clog.info(ComponentLogger.BOOT, f"Dry run: {args.dry_run}")
    if hasattr(args, 'heartbeat'):
        clog.info(ComponentLogger.BOOT, f"Heartbeat: {args.heartbeat}")
    
    # Hyperparameters
    clog.info(ComponentLogger.BOOT, f"Horizon: {args.horizon}")
    clog.info(ComponentLogger.BOOT, f"Entry score threshold: {args.entry_score_threshold}")
    clog.info(ComponentLogger.BOOT, f"Lambda DD: {args.lambda_dd}")
    clog.info(ComponentLogger.BOOT, f"Log level: {args.log_level}")
    
    # Base models
    clog.info(ComponentLogger.BOOT, f"Number of base models: {len(specs)}")
    for i, spec in enumerate(specs):
        clog.info(ComponentLogger.BOOT, f"  Model {i+1}: name={spec['name']} kind={spec['kind']}")
        clog.debug(ComponentLogger.IO, f"    script: {spec['script_path']}")
        clog.debug(ComponentLogger.IO, f"    ckpt: {spec['ckpt_path']}")
    
    clog.info(ComponentLogger.BOOT, "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meta-Policy Ensemble for Trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode arguments
    parser.add_argument("--live", action="store_true", help="Run in live trading mode")
    parser.add_argument("--train", action="store_true", help="Run in training mode (default if no mode specified)")
    
    # Model arguments
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        help="Comma-separated paths to model .py files",
    )
    parser.add_argument(
        "--ckpts",
        type=str,
        default=None,
        help="Comma-separated paths to model checkpoints",
    )
    parser.add_argument(
        "--meta_ckpt",
        type=str,
        default=None,
        help="Path to meta-policy checkpoint (for live mode)",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--horizon",
        type=int,
        default=META_HORIZON_BARS,
        help="Horizon bars for outcome computation",
    )
    parser.add_argument(
        "--entry_score_threshold",
        type=float,
        default=META_ENTRY_SCORE_THRESHOLD,
        help="Entry score threshold for NO_TRADE",
    )
    parser.add_argument(
        "--lambda_dd",
        type=float,
        default=META_LAMBDA_DD,
        help="Drawdown penalty lambda",
    )
    parser.add_argument(
        "--val_log_every_batches",
        type=int,
        default=META_VAL_LOG_BATCH_EVERY,
        help="Log validation progress every N batches (meta-policy). None disables per-batch val logging.",
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--log_every_seconds",
        type=float,
        default=1.0,
        help="Seconds between heartbeat logs in live mode",
    )
    
    # Live mode arguments
    parser.add_argument(
        "--heartbeat",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Enable heartbeat logging every second in live mode",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run all logic but do NOT send orders (still logs what would be sent)",
    )
    
    # Exit policy arguments
    parser.add_argument(
        "--train_exit_policy",
        action="store_true",
        help="Train the exit policy (requires entry policy to exist or be trained first)",
    )
    parser.add_argument(
        "--exit_policy_ckpt",
        type=str,
        default=None,
        help="Path to exit policy checkpoint (for live mode)",
    )
    parser.add_argument(
        "--exit_max_hold_bars",
        type=int,
        default=EXIT_MAX_HOLD_BARS,
        help="Maximum holding period (force exit after this many bars)",
    )
    parser.add_argument(
        "--exit_lambda_dd",
        type=float,
        default=EXIT_LAMBDA_DD,
        help="Drawdown penalty for exit scoring",
    )
    parser.add_argument(
        "--exit_label_margin",
        type=float,
        default=EXIT_LABEL_MARGIN,
        help="Margin below optimal score for acceptable exit label",
    )
    parser.add_argument(
        "--exit_score_min",
        type=float,
        default=EXIT_SCORE_MIN,
        help="Minimum score to label as EXIT",
    )
    parser.add_argument(
        "--exit_prob_threshold",
        type=float,
        default=EXIT_PROB_THRESHOLD,
        help="p_exit threshold for live execution",
    )
    parser.add_argument(
        "--exit_train_on_entry_policy",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Train exit policy only on entries where entry policy would trade",
    )
    parser.add_argument(
        "--exit_train_samples_limit",
        type=int,
        default=None,
        help="Limit number of exit training samples (for speed)",
    )
    parser.add_argument(
        "--live_rebuild_features_every_tick",
        action="store_true",
        help="In live mode, rebuild base-model features every loop (not just on closed bars)",
    )
    parser.add_argument(
        "--exit_val_entries_limit",
        type=int,
        default=None,
        help="Limit number of unique entries processed during exit validation PnL proxy (useful for large datasets)",
    )
    parser.add_argument(
        "--exit_log_every_batches",
        type=int,
        default=None,
        help="Log exit-training progress every N batches (default ~= total_batches/10). Use 1 to log each batch.",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Determine model specs
    if args.modules and args.ckpts:
        specs = parse_model_specs(args.modules, args.ckpts)
    else:
        specs = DEFAULT_MODEL_SPECS
        if clog:
            clog.info(ComponentLogger.BOOT, "Using default model specs")
    
    # Log boot information
    log_boot_info(args, specs)
    
    # Load adapters
    if clog:
        clog.info(ComponentLogger.BOOT, "Loading model adapters...")
    adapters = [ModelAdapter(spec) for spec in specs]
    if clog:
        clog.info(ComponentLogger.BOOT, f"Loaded {len(adapters)} adapters successfully")
    
    if args.train_exit_policy:
        # Exit policy training mode
        if clog:
            clog.info(ComponentLogger.BOOT, "Starting EXIT POLICY training mode...")
        
        # First, we need entry labels. Check if entry policy exists or train it
        ckpts = list(CHECKPOINT_DIR.glob("meta_epoch*.pt"))
        if not ckpts:
            if clog:
                clog.info(ComponentLogger.BOOT, "No entry policy found, training entry policy first...")
            train_meta_policy(
                adapters,
                horizon=args.horizon,
                entry_threshold=args.entry_score_threshold,
                lambda_dd=args.lambda_dd,
                val_log_every_batches=args.val_log_every_batches,
            )
        
        # Load feature data and compute entry labels
        if clog:
            clog.info(ComponentLogger.DATA, "Computing entry labels for exit training...")
        first_adapter = adapters[0]
        df, feature_cols = first_adapter.module._build_feature_frame()
        
        # Ensure high/low columns
        if "xau_high" not in df.columns:
            df["xau_high"] = df.get("high", df["xau_close"])
        if "xau_low" not in df.columns:
            df["xau_low"] = df.get("low", df["xau_close"])
        
        # Compute outcomes and labels for entry policy
        ret_up, mae_up, ret_dn, mae_dn = _compute_outcomes(df, args.horizon, args.lambda_dd)
        entry_labels = _compute_labels(ret_up, mae_up, ret_dn, mae_dn, args.entry_score_threshold, args.lambda_dd)
        
        # Train exit policy
        train_exit_policy(
            adapters,
            entry_labels,
            max_hold_bars=args.exit_max_hold_bars,
            lambda_dd=args.exit_lambda_dd,
            label_margin=args.exit_label_margin,
            score_min=args.exit_score_min,
            train_on_entry_policy=args.exit_train_on_entry_policy,
            samples_limit=args.exit_train_samples_limit,
            log_every_batches=args.exit_log_every_batches,
            val_entries_limit=args.exit_val_entries_limit,
        )
    
    elif args.live:
        # Live mode
        if args.meta_ckpt:
            meta_ckpt = Path(args.meta_ckpt)
        else:
            # Find latest checkpoint
            ckpts = list(CHECKPOINT_DIR.glob("meta_epoch*.pt"))
            if not ckpts:
                if clog:
                    clog.info(ComponentLogger.BOOT, "No meta checkpoint found, training first...")
                meta_ckpt = train_meta_policy(
                    adapters,
                    horizon=args.horizon,
                    entry_threshold=args.entry_score_threshold,
                    lambda_dd=args.lambda_dd,
                    val_log_every_batches=args.val_log_every_batches,
                )
            else:
                meta_ckpt = max(ckpts, key=lambda p: p.stat().st_mtime)
                if clog:
                    clog.info(ComponentLogger.IO, f"Using latest meta checkpoint: {meta_ckpt}")
        
        if not meta_ckpt.exists():
            if clog:
                clog.error(ComponentLogger.ERROR, f"Meta checkpoint not found: {meta_ckpt}")
            raise RuntimeError(f"Meta checkpoint not found: {meta_ckpt}")
        
        # Exit policy checkpoint
        exit_ckpt = None
        if args.exit_policy_ckpt:
            exit_ckpt = Path(args.exit_policy_ckpt)
        
        run_live(
            adapters,
            meta_ckpt,
            exit_ckpt=exit_ckpt,
            dry_run=args.dry_run,
            heartbeat=args.heartbeat,
            log_every_seconds=args.log_every_seconds,
            exit_prob_threshold=args.exit_prob_threshold,
            exit_max_hold_bars=args.exit_max_hold_bars,
            rebuild_features_every_tick=args.live_rebuild_features_every_tick,
        )
    else:
        # Training/backtest mode (default) - train entry policy
        if clog:
            clog.info(ComponentLogger.BOOT, "Starting ENTRY POLICY training mode...")
        train_meta_policy(
            adapters,
            horizon=args.horizon,
            entry_threshold=args.entry_score_threshold,
            lambda_dd=args.lambda_dd,
            val_log_every_batches=args.val_log_every_batches,
        )


if __name__ == "__main__":
    main()
