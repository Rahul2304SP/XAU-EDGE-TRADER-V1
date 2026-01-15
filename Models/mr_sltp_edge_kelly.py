#!/usr/bin/env python
"""
TREND/MR VARIANT: ACTIVE ENTRYPOINT = main_trend_mr

XAU Neural Trader (Trending vs Mean Reversion)
---------------------------------------------
Minute-bar trader variant with an interpretable feature set:
- cross-asset relationships (XAU vs XAG, XAU vs USD index)
- regime structure (trend vs mean-reversion)
- levels + channel position
- session and timing context

The model is supervised with multi-head outputs:
- regime (trend vs mean-reversion)
- trade gate + direction
- level outcome (bounce vs break)
"""

# TREND REGIME CONTROLLER ADDITIONS
# - Trend lookbacks/pooling: TREND_SHORT_LOOKBACK_BARS, TREND_LONG_LOOKBACK_BARS, TREND_LONG_POOL_K/MODE
# - Trend labels: TREND_HORIZON_BARS, TREND_LABEL_LAMBDA_DD, TREND_LABEL_BALANCE_MODE, TREND_LABEL_POS_RATE
# - Trend gate thresholds: TAU_* and TEMP_* globals, TrendGateModule in model forward
# - Regime controller in live loop (p_trend hysteresis) with trend strategy overrides

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
import warnings
import faulthandler
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

# QuantReg intentionally not used; channel fitting is fast OLS + residual quantiles.

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None

# Suppress deprecated SDPA context manager warning from PyTorch.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*sdp_kernel.*")

print("BOOT: start of script", flush=True)
faulthandler.enable()


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "Advanced Modelling" / "Final Execution Files" / "XAU_T_CNN_TRADER.py"
        if candidate.exists():
            return parent
    return here.parents[2]


# DATA FOLDER: point this to where your M1 CSVs live (defaults to publish/data)
DATA_FOLDER = Path(__file__).resolve().parents[2] / "data"

REPO_ROOT = _resolve_repo_root()
BASE_DIR = REPO_ROOT / "Advanced Modelling" / "Final Execution Files" / "T V MR Exit" / "Dynamic SLTP"/ "15 Min" / "MRSLTP_EDGE_DIST_KELLY_3"
OUTPUT_DIR = BASE_DIR / "Outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
FEATURE_CACHE_DIR = OUTPUT_DIR / "feature_cache"
FEATURE_CACHE_PATH = FEATURE_CACHE_DIR / "trend_mr_feature_cache.parquet"
FEATURE_CACHE_META = FEATURE_CACHE_DIR / "trend_mr_feature_cache_meta.json"
DATA_SCRAPER_DIR = REPO_ROOT / "Data Scraper"
M1_DATA_DIR = DATA_FOLDER  # use the DATA FOLDER path for M1 bars

TARGET_SYMBOL = "XAUUSD"
XAG_SYMBOL = "XAGUSD"
USD_INDEX_SYMBOL = "DX.f"
USD_INDEX_ALIASES = ("DX.f", "DXY", "USDIDX", "DXY.f")

# Optional run toggles
USE_SLOW_STREAM = True
RUN_SMOKE_TEST = False
CACHE_FEATURES = True
TRAIN_LIVE_FILL = True
LOG_TRAIN_PROGRESS_EVERY = 50
LIVE_POLL_SECONDS = 1
LIVE_LOTS = 0.13
LIVE_SLIPPAGE = 20
LIVE_MAGIC = int(os.getenv("MT5_LIVE_MAGIC", "1001"))
LIVE_COMMENT = "15MR EDGE DIST KELLY 3"
EXIT_CONFIRM_BARS = 1
LIVE_LOG_FILENAME = "MR_STLP_EXIT_15_Trending_live_log.csv"
SPREAD_POINTS_TO_PRICE = 0.01
ENABLE_LIVE_FEATURE_TIMING = True
LIVE_FEATURE_PROGRESS = True
LIVE_FEATURE_PROGRESS_EVERY = 200
LIVE_P_UP_MIN = 0.55
LIVE_P_UP_SHORT_MAX = 0.45
# === ER VETO (MR ENTRY GUARD) ===
ER_WINDOW_MINUTES = 5
ER_BLOCK_ABS_THR = 0.75
ER_BLOCK_REQUIRE_CLOSED_BAR = True
ER_DEBUG_LOG = True

# Startup calibration for entry thresholds
CALIBRATE_THRESHOLDS_ON_STARTUP = True
CALIBRATION_PERCENTILE = 75.0
CALIBRATION_BARS = 5000
CALIBRATION_MIN_SAMPLES_PER_REGIME = 200
CALIBRATION_SAVE_JSON = True
CALIBRATION_JSON_PATH = str(OUTPUT_DIR / "startup_thresholds.json")
CALIBRATION_THR_FLOOR = 0.0
CALIBRATION_THR_CEIL = 1.0
CALIBRATION_DEFAULT_REGIME = "TREND"
CALIBRATION_USE_FULL_DATASET = True
CALIBRATION_USE_FEATURE_CACHE = True

# MT5 terminal routing (set to specific terminal64.exe to target that instance)
MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH", "")

# Dynamic lot sizing bounds
LOT_MIN = 0.15
LOT_MAX = 0.75

# --- Dynamic SL/TP toggles ---
ENABLE_SL = True
ENABLE_TP = True
ENABLE_TRAILING = True

# ATR settings (classic ATR on M1)
ATR_PERIOD = 15
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0

# --- Learned SL/TP (Phase 1) ---
LEARNED_SLTP = True
APPLY_K_CLAMPS = True
CLAMP_LABELS_TOO = True

# Bounds (used only when APPLY_K_CLAMPS / CLAMP_LABELS_TOO are True)
K_SL_MIN = 0.60
K_SL_MAX = 3.00
K_TP_MIN = 0.80
K_TP_MAX = 5.00

SLTP_LABEL_HORIZON_BARS = 30

LOSS_W_KSL = 0.30
LOSS_W_KTP = 0.20

# --- Extra SL widening (absolute dollars) ---
SL_EXTRA_DOLLARS = 1.00
MAX_SL_EXTRA_ATR = 1.00

# --- Dynamic ATR multiplier mapping (Option A) ---
DYNAMIC_ATR_MULTS = True

# Hard clamps (professional safety rails)
SL_ATR_MULT_MIN = 1.00
SL_ATR_MULT_MAX = 3.50
TP_ATR_MULT_MIN = 1.00
TP_ATR_MULT_MAX = 4.00

# Uncertainty penalty configuration
UNCERTAINTY_NORM = 0.30
UNCERTAINTY_PENALTY = 1.50

# Optional: dynamic trailing behavior derived from k_sl
DYNAMIC_TRAIL_MULTS = True
TRAIL_ATR_MULT_MIN = 1
TRAIL_ATR_MULT_MAX = 2.00
TRAIL_START_ATR_MIN = 0.7
TRAIL_START_ATR_MAX = 1.6

# Trailing behavior
TRAIL_START_ATR = 1.0
TRAIL_ATR_MULT = 1.2
TRAIL_UPDATE_MIN_STEP_ATR = 0.10

# --- Breakout head (Option A1) ---
ENABLE_BREAKOUT_HEAD = True
LOSS_W_BREAK = 0.20
POS_W_BREAK = 1.0
BREAK_HORIZON_BARS = 30
BREAK_GO_ATR = 1.0
BREAK_FAIL_RETRACE_ATR = 0.6

# Exceedance features (close-based)
EXCEED_USE_CLOSE = True
EXCEED_EPS = 1e-8

# MC Dropout uncertainty (live only)
ENABLE_MC_DROPOUT_UNCERTAINTY = True
MC_DROPOUT_PASSES = 16

# ---- Time-to-Profit (TTP) head config ----
USE_TTP_HEAD = True
TTP_ATR_MULT = 0.30
TTP_MAX_MINUTES = 2
TTP_HORIZON_BARS = None
TTP_LABEL_MODE = "binary_quick"
TTP_LABEL_SMOOTH = 0.02
TTP_REQUIRE_DIRECTION = True
TTP_USE_ENTRY_ONLY_MASK = True
TTP_DIR_LABEL_SOURCE = "existing_labels"
TTP_TRAIN_ON_Y_TRADE_ONLY = True
TTP_MIN_Y_TRADE = 0.5
LOSS_W_TTP = 1.0
TTP_USE_AS_GATE = True
TTP_MIN_PROB = 0.60
TTP_GATE_MODE = "soft"
LOG_TTP_STATS_EVERY = 500
EXPORT_TTP_TO_CSV = True
TTP_HEAD_HIDDEN = 32
TTP_HEAD_DROPOUT = 0.10

# --- Edge / Expected Return head ---
USE_EDGE_HEAD = True
EDGE_HORIZON_BARS = 15
EDGE_RET_UNIT = "ATR"
EDGE_ATR_EPS = 1e-8
EDGE_CLIP_RET_ATR = 5.0
EDGE_TRAIN_ON_Y_TRADE_ONLY = True
EDGE_DIR_LABEL_SOURCE = "existing_labels"
EDGE_HEAD_HIDDEN = 32
EDGE_HEAD_DROPOUT = 0.10
LOSS_W_EDGE = 0.40

# --- Distributional / Quantile head (tail-aware) ---
USE_QDIST_HEAD = True
QDIST_HORIZON_BARS = 15
QDIST_QUANTILES = (0.1, 0.5, 0.9)
QDIST_RET_UNIT = "ATR"
QDIST_CLIP_RET_ATR = 6.0
QDIST_TRAIN_ON_Y_TRADE_ONLY = True
LOSS_W_QDIST = 0.60
QDIST_LOSS = "pinball"
QDIST_TAIL_EXTRA_WEIGHT = 2.0
QDIST_MONOTONIC_PENALTY_W = 0.10

# --- Risk-aware objective (downside penalized more) ---
USE_RISK_AWARE_LOSS = True
DOWNSIDE_MULT = 2.0
UPSIDE_MULT = 1.0
RISK_AWARE_MODE = "both"
RISK_AWARE_HUBER_DELTA = 1.0

# --- Kelly sizing ---
USE_KELLY_SIZING = True
KELLY_MODE = "prob_and_payoff"
KELLY_FRACTIONAL = 0.25
KELLY_CLIP_MIN = 0.0
KELLY_CLIP_MAX = 0.5
KELLY_MIN_LOT = 0.15
KELLY_MAX_LOT = 1.00
KELLY_LOT_STEP = 0.01
KELLY_USE_P_TRADE_EFF = True
KELLY_USE_TAIL_RISK = True
KELLY_TAIL_RISK_LAMBDA = 1.0

# --- Adaptive thresholds for p_trade_eff ---
USE_ADAPTIVE_EFF_THR = True
EFF_THR_MODE = "rolling_percentile"
EFF_THR_ROLL_N = 2000
EFF_THR_PERCENTILE_TREND = 0.75
EFF_THR_PERCENTILE_MR = 0.80
EFF_THR_MIN = 0.05
EFF_THR_MAX = 0.95
EFF_THR_WARMUP_BARS = 200
EFF_THR_STATIC_TREND = 0.55
EFF_THR_STATIC_MR = 0.60
EFF_THR_UPDATE_EVERY = 5
EFF_THR_DEBUG_EVERY = 50
# --- Multi-horizon slope windows (for disagreement / transition) ---
SLOPE_WINDOWS = (2, 4, 8, 16, 32, 64, 128)
SLOPE_USE_ATR_NORM = True
SLOPE_ATR_EPS = 1e-6

# Disagreement feature config
DISAGREE_USE_SIGN = True
DISAGREE_USE_MAG = True
DISAGREE_EPS = 1e-8
DISAGREE_CLIP = 5.0

# --- Transition head config ---
USE_TRANSITION_HEAD = True
TRANSITION_HEAD_HIDDEN = 32
TRANSITION_HEAD_DROPOUT = 0.10
LOSS_W_TRANSITION = 0.60
POS_W_TRANSITION = 2.0
USE_POS_W_TRANSITION = True
TRANSITION_LABEL_HORIZON_BARS = 15
TRANSITION_LABEL_ATR_MOVE = 0.30
TRANSITION_LABEL_MODE = "direction_takeover"
TRANSITION_LABEL_SMOOTH = 0.02

# Transition mode behavior (live)
TRANSITION_USE_IN_TREND_MODE = True
TRANSITION_MIN_PROB = 0.60
TRANSITION_DIR_SOURCE = "short_vs_long"
TRANSITION_REQUIRE_TREND = True
TRANSITION_TREND_MIN = 0.55
TRANSITION_MIN_STRENGTH = 0.10
TRANSITION_DEBUG_EVERY_BARS = 30

# Transition-driven exit behavior
TRANSITION_EXIT_ENABLE = True
TRANSITION_EXIT_MIN_PROB = 0.65
TRANSITION_EXIT_REQUIRE_IN_TREND = True
TRANSITION_EXIT_TREND_MIN = 0.55
TRANSITION_EXIT_MIN_BARS_IN_TRADE = 1
TRANSITION_EXIT_ACTION = "close"
TRANSITION_EXIT_TIGHTEN_ATR = 0.25
TRANSITION_EXIT_CONFIRM_BARS = 1

# Transition head inputs
TRANSITION_INCLUDE_SHARED = True
TRANSITION_INCLUDE_SLOPES = True
TRANSITION_INCLUDE_DISAGREE = True

# --- Trend regime controller ---
TREND_SHORT_LOOKBACK_BARS = 130
TREND_LONG_LOOKBACK_BARS = 450
TREND_LONG_POOL_K = 5
TREND_LONG_POOL_MODE = "mean"
TREND_SHORT_PUSH_WINDOW = 30
TREND_EPS = 1e-6

TREND_HORIZON_BARS = 15
TREND_LABEL_LAMBDA_DD = 0.7
TREND_LABEL_BALANCE_MODE = "quantile"
TREND_LABEL_POS_RATE = 0.30

LOSS_W_TREND = 1.0
LOSS_W_TREND_DIR = 0.5
POS_W_TREND = 3.0
TREND_DIR_MIN_STRENGTH = 0.2
TREND_DIR_ADV_TEMP = 0.5
TREND_DIR_ADV_CONF_TEMP = 0.5

TAU_SLOPE_INIT = 0.02
TAU_PUSH_INIT = 0.8
TAU_PULL_INIT = 0.8
TAU_RATIO_INIT = 1.5
TAU_MODE_INIT = 0.5
TEMP_SLOPE = 0.5
TEMP_PUSH = 0.5
TEMP_PULL = 0.5
TEMP_RATIO = 0.5
TEMP_MODE = 0.5
TEMP_DIR = 0.5

TREND_ENTER_THR = 0.65
TREND_EXIT_THR = 0.45
TREND_PERSIST_BARS = 2
REGIME_BLEND_MODE = False
TREND_REQUIRE_P_TRADE = True

TREND_KSL_MULT = 1.2
TREND_KTP_MULT = 1.5
TREND_TRAIL_START_ATR = 0.8
TREND_TRAIL_ATR_MULT = 1.0
TREND_DIAG_EVERY_BARS = 30

TREND_VOL_COL_NAME = "vol_30m"
TREND_VOL_COL_IDX: Optional[int] = None
DEBUG_TREND_GATE_LIVE = True
DEBUG_TREND_GATE_EVERY_N_BARS = 1

# --- Diagnostic export ---
DIAG_EXPORT = False
DIAG_EXPORT_DIR = "C:/temp/diag_exports"
DIAG_EXPORT_MAX_ROWS = 200000
DIAG_EXPORT_SPLIT = "both"
DIAG_EXPORT_EVERY = 1
DIAG_EXPORT_FORMAT = "npz"

# --- Learned entry combiner (Option E) ---
USE_LEARNED_ENTRY_COMBINER = True
COMBINER_USE_LOGITS = True
COMBINER_INCLUDE_CONTEXT = True
COMBINER_INCLUDE_TTP = True
COMBINER_TTP_CLIP = 6.0
COMBINER_INCLUDE_EDGE = True
COMBINER_INCLUDE_QDIST = True
COMBINER_INCLUDE_TAIL = True
COMBINER_EDGE_CLIP = 6.0
COMBINER_QDIST_CLIP = 6.0
COMBINER_TAIL_CLIP = 6.0
COMBINER_HIDDEN = 32
COMBINER_DROPOUT = 0.10
LOSS_W_COMBINER = 1.0
COMBINER_TARGET_MODE = "trade_and_ttp"
COMBINER_TTP_FLOOR = 0.20
COMBINER_USE_SOFT_TARGET = True
COMBINER_TRAIN_MASK_MODE = "trade_and_ttp_mask"
COMBINER_DIR_MARGIN_ATR = 0.25
COMBINER_LABEL_SMOOTH = 0.02
LIVE_TRADE_EFF_THR_MR = 0.60
LIVE_TRADE_EFF_THR_TREND = 0.55
COMBINER_FALLBACK_MODE = "p_trade"

# Feature configuration
RET_WINDOWS = (5,)
VOL_WINDOW = 5
CORR_WINDOWS = (6, 10)

LEVEL_LOOKBACK_DAYS = 7
LEVEL_K = 12
LEVEL_REBUILD_HOURS = 4
LEVEL_TOUCH_HOURS = 4
LEVEL_TOL_PCT = 0.001
LEVEL_BOUNCE_MULT = 2.0
LEVEL_BREAK_MULT = 1.0

CHANNEL_WINDOW_MINUTES = 240
CHANNEL_REFIT_MINUTES = 5
CHANNEL_Q_LOWER = 0.1
CHANNEL_Q_MID = 0.5
CHANNEL_Q_UPPER = 0.9

SESSION_MEDIAN_DAYS = 5
SESSION_OPEN_WINDOW_MIN = 60 #Used to be 30

# Live feature timing helpers (current bar only)
_TIMING_STARTS: dict[str, float] = {}
_TIMING_MS: dict[str, float] = {}
_TIMING_ACTIVE = False
_CORR_WARNED: set[str] = set()


def start_timer(name: str) -> None:
    if not ENABLE_LIVE_FEATURE_TIMING or not _TIMING_ACTIVE:
        return
    _TIMING_STARTS[name] = time.perf_counter()


def stop_timer(name: str) -> None:
    if not ENABLE_LIVE_FEATURE_TIMING or not _TIMING_ACTIVE:
        return
    start = _TIMING_STARTS.get(name)
    if start is None:
        return
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _TIMING_MS[name] = _TIMING_MS.get(name, 0.0) + elapsed_ms


def _reset_timing() -> None:
    _TIMING_STARTS.clear()
    _TIMING_MS.clear()


def _log_timing(writer: Optional[SummaryWriter], step: int) -> None:
    if not ENABLE_LIVE_FEATURE_TIMING or not _TIMING_MS:
        return
    try:
        total_ms = int(round(_TIMING_MS.get("total", 0.0)))
        keys = [
            "fetch_data",
            "feature_frame",
            "fast_features",
            "correlations",
            "volatility",
            "channel",
            "levels",
            "sequence_build",
            "model_inference",
        ]
        items = [(k, int(round(_TIMING_MS.get(k, 0.0)))) for k in keys]
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        parts = " | ".join(f"{k}={v}ms" for k, v in items_sorted)
        logging.info("[FEATURE_TIMING] total=%sms | %s", total_ms, parts)
        if writer is not None:
            writer.add_scalar("live_timing/total_ms", total_ms, step)
            for key, val in items:
                writer.add_scalar(f"live_timing/{key}_ms", val, step)
    except Exception:
        logging.exception("Timing log failed")

# Training configuration
HORIZON_MINUTES = 15
MAX_HOLD_MINUTES = 15
MR_TRADE_THRESHOLD = 0.65
TREND_TRADE_THRESHOLD = 0.032
TRADE_THRESHOLD = MR_TRADE_THRESHOLD
DIR_LOSS_WEIGHT = 1.0
LOSS_W_TRADE = 1.0
LOSS_W_DIR = 1.5
LOSS_W_BOUNCE = 0.4
POS_W_TRADE = 3.0
POS_W_DIR = 1.0
POS_W_BOUNCE = 1.0

# --- Distributional exit head (future return curve) ---
DIST_EXIT_ENABLE = True
DIST_EXIT_HORIZONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
DIST_EXIT_QUANTILES = (5, 10, 25, 50, 75, 95)
DIST_EXIT_LOSS_W = 0.30
DIST_EXIT_CLAMP_MIN = -5.0
DIST_EXIT_CLAMP_MAX = 5.0
DIST_EXIT_CONFIRM_BARS = 2
DIST_EXIT_ONLY_IF_POSITIVE_PNL = True
DIST_EXIT_MIN_BARS_IN_TRADE = 2
DIST_EXIT_USE_EV = True
DIST_EXIT_EV_WEIGHTS = (0.25, 0.50, 0.25)
DIST_EXIT_EV_HORIZON = 5
DIST_EXIT_EV_THRESHOLD = 0.05
DIST_EXIT_DOWNSIDE_LIMIT = -0.25
DIST_EXIT_MEDIAN_FLAT_EPS = 0.02

TRAIN_SPLIT = 0.8
TRAIN_START_IDX = None
TRAIN_END_IDX = None
VAL_START_IDX = None
VAL_END_IDX = None
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
GRAD_CLIP_NORM = 1.0

# Model configuration
SHORT_LEN = 15
MID_LEN = 60
LONG_LEN = 240
SLOW_FREQ_MINUTES = 5
SLOW_LEN = 1000

EMBED_DIM = 128
LAYERS = 3
NHEAD = 8
DROPOUT = 0.05
TCN_CHANNELS = 64
TCN_KERNEL = 15
TCN_LAYERS = 3

DEVICE = torch.device("cuda")

# Trend/MR toggles for clarity (only the new feature families are enabled)
FEATURE_TOGGLES_BASE_OVERRIDE = {
    "USE_CROSS_ASSET": True,
    "USE_CORR_BETA": True,
    "USE_XAUCORE": True,
    "USE_LEVELS_KMEANS": True,
    "USE_CHANNEL_QUANTREG": True,
    "USE_SESSION_FLAGS": True,
    "use_slow_stream": True,
    "USE_HMM_REGIME_FEATURES": False,
    "USE_NONPARAM_FEATURES": False,
    "USE_TICK_FEATURES": False,
    "USE_RL_POLICY": False,
    "USE_EXIT_HEAD": False,
}

OVERRIDE_GLOBALS = {
    "TARGET_SYMBOL": TARGET_SYMBOL,
    "XAG_SYMBOL": XAG_SYMBOL,
    "USD_INDEX_SYMBOL": USD_INDEX_SYMBOL,
    "USE_SLOW_STREAM": bool(FEATURE_TOGGLES_BASE_OVERRIDE.get("use_slow_stream", True)),
    "MT5_TERMINAL_PATH": MT5_TERMINAL_PATH,
    "RUN_SMOKE_TEST": RUN_SMOKE_TEST,
    "CACHE_FEATURES": CACHE_FEATURES,
    "SPREAD_POINTS_TO_PRICE": SPREAD_POINTS_TO_PRICE,
    "RET_WINDOWS": RET_WINDOWS,
    "VOL_WINDOW": VOL_WINDOW,
    "CORR_WINDOWS": CORR_WINDOWS,
    "LEVEL_LOOKBACK_DAYS": LEVEL_LOOKBACK_DAYS,
    "LEVEL_K": LEVEL_K,
    "LEVEL_REBUILD_HOURS": LEVEL_REBUILD_HOURS,
    "LEVEL_TOUCH_HOURS": LEVEL_TOUCH_HOURS,
    "LEVEL_TOL_PCT": LEVEL_TOL_PCT,
    "BOUNCE_THRESH_PCT": LEVEL_BOUNCE_MULT,
    "BREAK_THRESH_PCT": LEVEL_BREAK_MULT,
    "CHANNEL_WINDOW_MIN": CHANNEL_WINDOW_MINUTES,
    "CHANNEL_REFIT_EVERY_MIN": CHANNEL_REFIT_MINUTES,
    "Q_LO": CHANNEL_Q_LOWER,
    "Q_MID": CHANNEL_Q_MID,
    "Q_HI": CHANNEL_Q_UPPER,
    "SESSION_MEDIAN_DAYS": SESSION_MEDIAN_DAYS,
    "TARGET_HORIZON_MIN": HORIZON_MINUTES,
    "TRADE_THRESHOLD": TRADE_THRESHOLD,
    "EPOCHS": EPOCHS,
    "LR": LEARNING_RATE,
    "BATCH_SIZE": BATCH_SIZE,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "GRAD_CLIP_NORM": GRAD_CLIP_NORM,
    "DIR_LOSS_WEIGHT": DIR_LOSS_WEIGHT,
    "BOUNCE_LOSS_WEIGHT": LOSS_W_BOUNCE,
    "SHORT_LEN": SHORT_LEN,
    "MID_LEN": MID_LEN,
    "LONG_LEN": LONG_LEN,
    "EMBED_DIM": EMBED_DIM,
    "LAYERS": LAYERS,
    "NHEAD": NHEAD,
    "DROPOUT": DROPOUT,
    "TCN_CHANNELS": TCN_CHANNELS,
    "TCN_KERNEL": TCN_KERNEL,
    "TCN_LAYERS": TCN_LAYERS,
    "SLOW_FREQ_MINUTES": SLOW_FREQ_MINUTES,
    "SLOW_LEN": SLOW_LEN,
    "TRAIN_START_IDX": TRAIN_START_IDX,
    "TRAIN_END_IDX": TRAIN_END_IDX,
    "VAL_START_IDX": VAL_START_IDX,
    "VAL_END_IDX": VAL_END_IDX,
}

ORIGINAL_LINE_COUNT = 3787

# Feature toggle flags (defaults from FEATURE_TOGGLES_BASE_OVERRIDE)
USE_CROSS_ASSET = bool(FEATURE_TOGGLES_BASE_OVERRIDE.get("USE_CROSS_ASSET", True))
USE_CORR_BETA = bool(FEATURE_TOGGLES_BASE_OVERRIDE.get("USE_CORR_BETA", True))
USE_XAUCORE = bool(FEATURE_TOGGLES_BASE_OVERRIDE.get("USE_XAUCORE", True))
USE_LEVELS_KMEANS = bool(FEATURE_TOGGLES_BASE_OVERRIDE.get("USE_LEVELS_KMEANS", True))
USE_CHANNEL_QUANTREG = bool(FEATURE_TOGGLES_BASE_OVERRIDE.get("USE_CHANNEL_QUANTREG", True))
USE_SESSION_FLAGS = bool(FEATURE_TOGGLES_BASE_OVERRIDE.get("USE_SESSION_FLAGS", True))


def _apply_overrides() -> dict:
    applied_globals: list[str] = []
    applied_toggles: list[str] = []
    warnings: list[str] = []

    def _set_global(name: str, value: object) -> None:
        globals()[name] = value

    def _ensure_seq(val: object) -> tuple:
        if isinstance(val, (list, tuple)):
            return tuple(val)
        return (val,)

    toggle_map = {
        "USE_CROSS_ASSET": "USE_CROSS_ASSET",
        "USE_CORR_BETA": "USE_CORR_BETA",
        "USE_XAUCORE": "USE_XAUCORE",
        "USE_LEVELS_KMEANS": "USE_LEVELS_KMEANS",
        "USE_CHANNEL_QUANTREG": "USE_CHANNEL_QUANTREG",
        "USE_SESSION_FLAGS": "USE_SESSION_FLAGS",
        "USE_SLOW_STREAM": "USE_SLOW_STREAM",
        "use_slow_stream": "USE_SLOW_STREAM",
    }
    for key, val in FEATURE_TOGGLES_BASE_OVERRIDE.items():
        if key in toggle_map:
            _set_global(toggle_map[key], bool(val))
            applied_toggles.append(key)
    if "use_slow_stream" in FEATURE_TOGGLES_BASE_OVERRIDE:
        _set_global("USE_SLOW_STREAM", bool(FEATURE_TOGGLES_BASE_OVERRIDE["use_slow_stream"]))
        applied_toggles.append("use_slow_stream")
    elif "USE_SLOW_STREAM" in FEATURE_TOGGLES_BASE_OVERRIDE:
        _set_global("USE_SLOW_STREAM", bool(FEATURE_TOGGLES_BASE_OVERRIDE["USE_SLOW_STREAM"]))
        applied_toggles.append("USE_SLOW_STREAM")

    for key, val in OVERRIDE_GLOBALS.items():
        if key == "RET_WINDOWS":
            _set_global("RET_WINDOWS", _ensure_seq(val))
            applied_globals.append(key)
        elif key == "VOL_WINDOW":
            _set_global("VOL_WINDOW", int(val))
            applied_globals.append(key)
        elif key == "CORR_WINDOWS":
            _set_global("CORR_WINDOWS", _ensure_seq(val))
            applied_globals.append(key)
        elif key == "LR":
            _set_global("LEARNING_RATE", float(val))
            applied_globals.append(key)
        elif key == "GRAD_CLIP_NORM":
            _set_global("GRAD_CLIP_NORM", float(val))
            applied_globals.append(key)
        elif key == "DIR_LOSS_WEIGHT":
            _set_global("DIR_LOSS_WEIGHT", float(val))
            applied_globals.append(key)
        elif key == "BOUNCE_LOSS_WEIGHT":
            _set_global("LOSS_W_BOUNCE", float(val))
            applied_globals.append(key)
        elif key == "TARGET_HORIZON_MIN":
            _set_global("HORIZON_MINUTES", int(val))
            applied_globals.append(key)
        elif key == "CHANNEL_WINDOW_MIN":
            _set_global("CHANNEL_WINDOW_MINUTES", int(val))
            applied_globals.append(key)
        elif key == "CHANNEL_REFIT_EVERY_MIN":
            _set_global("CHANNEL_REFIT_MINUTES", int(val))
            applied_globals.append(key)
        elif key == "Q_LO":
            _set_global("CHANNEL_Q_LOWER", float(val))
            applied_globals.append(key)
        elif key == "Q_MID":
            _set_global("CHANNEL_Q_MID", float(val))
            applied_globals.append(key)
        elif key == "Q_HI":
            _set_global("CHANNEL_Q_UPPER", float(val))
            applied_globals.append(key)
        elif key == "BOUNCE_THRESH_PCT":
            _set_global("LEVEL_BOUNCE_MULT", float(val))
            applied_globals.append(key)
        elif key == "BREAK_THRESH_PCT":
            _set_global("LEVEL_BREAK_MULT", float(val))
            applied_globals.append(key)
        elif key == "SPREAD_POINTS_TO_PRICE":
            _set_global("SPREAD_POINTS_TO_PRICE", float(val))
            applied_globals.append(key)
        elif key in globals():
            _set_global(key, val)
            applied_globals.append(key)
        elif key in toggle_map:
            _set_global(toggle_map[key], bool(val))
            applied_toggles.append(key)
        else:
            warnings.append(key)

    config = {
        "TARGET_SYMBOL": TARGET_SYMBOL,
        "XAG_SYMBOL": XAG_SYMBOL,
        "USD_INDEX_SYMBOL": USD_INDEX_SYMBOL,
        "RET_WINDOWS": RET_WINDOWS,
        "VOL_WINDOW": VOL_WINDOW,
        "CORR_WINDOWS": CORR_WINDOWS,
        "LEVEL_LOOKBACK_DAYS": LEVEL_LOOKBACK_DAYS,
        "LEVEL_K": LEVEL_K,
        "LEVEL_REBUILD_HOURS": LEVEL_REBUILD_HOURS,
        "LEVEL_TOUCH_HOURS": LEVEL_TOUCH_HOURS,
        "LEVEL_TOL_PCT": LEVEL_TOL_PCT,
        "LEVEL_BOUNCE_MULT": LEVEL_BOUNCE_MULT,
        "LEVEL_BREAK_MULT": LEVEL_BREAK_MULT,
        "CHANNEL_WINDOW_MINUTES": CHANNEL_WINDOW_MINUTES,
        "CHANNEL_REFIT_MINUTES": CHANNEL_REFIT_MINUTES,
        "CHANNEL_Q_LOWER": CHANNEL_Q_LOWER,
        "CHANNEL_Q_MID": CHANNEL_Q_MID,
        "CHANNEL_Q_UPPER": CHANNEL_Q_UPPER,
        "SESSION_MEDIAN_DAYS": SESSION_MEDIAN_DAYS,
        "HORIZON_MINUTES": HORIZON_MINUTES,
        "TRADE_THRESHOLD": TRADE_THRESHOLD,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "GRAD_CLIP_NORM": GRAD_CLIP_NORM,
        "DIR_LOSS_WEIGHT": DIR_LOSS_WEIGHT,
        "LOSS_W_TRADE": LOSS_W_TRADE,
        "LOSS_W_DIR": LOSS_W_DIR,
        "LOSS_W_BOUNCE": LOSS_W_BOUNCE,
        "SHORT_LEN": SHORT_LEN,
        "MID_LEN": MID_LEN,
        "LONG_LEN": LONG_LEN,
        "EMBED_DIM": EMBED_DIM,
        "LAYERS": LAYERS,
        "NHEAD": NHEAD,
        "DROPOUT": DROPOUT,
        "TCN_CHANNELS": TCN_CHANNELS,
        "TCN_KERNEL": TCN_KERNEL,
        "TCN_LAYERS": TCN_LAYERS,
        "SLOW_FREQ_MINUTES": SLOW_FREQ_MINUTES,
        "SLOW_LEN": SLOW_LEN,
        "TRAIN_START_IDX": TRAIN_START_IDX,
        "TRAIN_END_IDX": TRAIN_END_IDX,
        "VAL_START_IDX": VAL_START_IDX,
        "VAL_END_IDX": VAL_END_IDX,
        "USE_SLOW_STREAM": USE_SLOW_STREAM,
        "MT5_TERMINAL_PATH": MT5_TERMINAL_PATH,
        "RUN_SMOKE_TEST": RUN_SMOKE_TEST,
        "CACHE_FEATURES": CACHE_FEATURES,
        "SPREAD_POINTS_TO_PRICE": SPREAD_POINTS_TO_PRICE,
        "SPREAD_POINTS_TO_PRICE": SPREAD_POINTS_TO_PRICE,
        "feature_toggles": {
            "USE_CROSS_ASSET": USE_CROSS_ASSET,
            "USE_CORR_BETA": USE_CORR_BETA,
            "USE_XAUCORE": USE_XAUCORE,
            "USE_LEVELS_KMEANS": USE_LEVELS_KMEANS,
            "USE_CHANNEL_QUANTREG": USE_CHANNEL_QUANTREG,
            "USE_SESSION_FLAGS": USE_SESSION_FLAGS,
            "USE_SLOW_STREAM": USE_SLOW_STREAM,
        },
        "applied_globals": applied_globals,
        "applied_toggles": applied_toggles,
        "warnings": warnings,
    }
    return config


def _effective_config() -> dict:
    return {
        "USE_SLOW_STREAM": USE_SLOW_STREAM,
        "MT5_TERMINAL_PATH": MT5_TERMINAL_PATH,
        "SHORT_LEN": SHORT_LEN,
        "MID_LEN": MID_LEN,
        "LONG_LEN": LONG_LEN,
        "EMBED_DIM": EMBED_DIM,
        "LAYERS": LAYERS,
        "NHEAD": NHEAD,
        "DROPOUT": DROPOUT,
        "TCN_CHANNELS": TCN_CHANNELS,
        "TCN_KERNEL": TCN_KERNEL,
        "TCN_LAYERS": TCN_LAYERS,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "GRAD_CLIP_NORM": GRAD_CLIP_NORM,
        "HORIZON_MINUTES": HORIZON_MINUTES,
        "TRADE_THRESHOLD": TRADE_THRESHOLD,
        "LEVEL_K": LEVEL_K,
        "LEVEL_LOOKBACK_DAYS": LEVEL_LOOKBACK_DAYS,
        "LEVEL_REBUILD_HOURS": LEVEL_REBUILD_HOURS,
        "LEVEL_TOL_PCT": LEVEL_TOL_PCT,
        "BOUNCE_THRESH_PCT": LEVEL_BOUNCE_MULT,
        "BREAK_THRESH_PCT": LEVEL_BREAK_MULT,
        "CHANNEL_WINDOW_MIN": CHANNEL_WINDOW_MINUTES,
        "CHANNEL_REFIT_EVERY_MIN": CHANNEL_REFIT_MINUTES,
        "Q_LO": CHANNEL_Q_LOWER,
        "Q_MID": CHANNEL_Q_MID,
        "Q_HI": CHANNEL_Q_UPPER,
        "RET_WINDOWS": RET_WINDOWS,
        "CORR_WINDOWS": CORR_WINDOWS,
        "VOL_WINDOW": VOL_WINDOW,
        "TRAIN_START_IDX": TRAIN_START_IDX,
        "TRAIN_END_IDX": TRAIN_END_IDX,
        "VAL_START_IDX": VAL_START_IDX,
        "VAL_END_IDX": VAL_END_IDX,
        "SPREAD_POINTS_TO_PRICE": SPREAD_POINTS_TO_PRICE,
    }

def _ensure_output_dirs() -> None:
    for path in (BASE_DIR, OUTPUT_DIR, RUNS_DIR, CHECKPOINT_DIR, METRICS_DIR, FEATURE_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _abort_nonfinite(reason: str, payload: dict) -> None:
    path = OUTPUT_DIR / f"trend_mr_nonfinite_{reason}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        path.write_text(json.dumps(payload, indent=2, default=str))
    except Exception:
        pass
    os._exit(1)


def _to_utc_series(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC")
    return dt


def _normalize_bar_dt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bar_dt"] = _to_utc_series(df["bar_dt"])
    df = df.dropna(subset=["bar_dt"]).sort_values("bar_dt")
    df["bar_dt"] = df["bar_dt"].dt.floor("min")
    df = df.drop_duplicates(subset=["bar_dt"], keep="last")
    return df.reset_index(drop=True)


M1_CACHE_MT5: dict[str, pd.DataFrame] = {}
M1_CACHE_CSV: dict[str, pd.DataFrame] = {}

def _mt5_initialize() -> bool:
    if mt5 is None:
        return False
    try:
        mt5.shutdown()
    except Exception:
        pass
    if MT5_TERMINAL_PATH:
        return mt5.initialize(path=MT5_TERMINAL_PATH)
    return mt5.initialize()


def load_m1_from_mt5(symbol: str, max_bars: int = 500000) -> pd.DataFrame:
    if symbol in M1_CACHE_MT5:
        return M1_CACHE_MT5[symbol].copy()
    if mt5 is None:
        return pd.DataFrame(columns=["bar_dt", "open", "high", "low", "close", "volume", "spread"])
    if not _mt5_initialize():
        logging.info("MT5 initialize failed")
        return pd.DataFrame(columns=["bar_dt", "open", "high", "low", "close", "volume", "spread"])

    rates: list[dict] = []
    batch = 5000
    pos = 0
    while pos < max_bars:
        chunk = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, pos, batch)
        if chunk is None or len(chunk) == 0:
            break
        rates.extend(chunk)
        pos += batch
    mt5.shutdown()

    if not rates:
        return pd.DataFrame(columns=["bar_dt", "open", "high", "low", "close", "volume", "spread"])

    df = pd.DataFrame(rates)
    if "time" in df.columns:
        df.rename(columns={"time": "bar_dt"}, inplace=True)
    elif "bar_dt" not in df.columns:
        return pd.DataFrame(columns=["bar_dt", "open", "high", "low", "close", "volume", "spread"])
    df["bar_dt"] = pd.to_datetime(df["bar_dt"], unit="s", utc=True, errors="coerce")
    df["volume"] = df.get("tick_volume", df.get("real_volume", 0.0))
    df["spread"] = pd.to_numeric(df.get("spread", 0.0), errors="coerce").fillna(0.0)
    point = None
    try:
        info = mt5.symbol_info(symbol)
        if info is not None and getattr(info, "point", None):
            point = float(info.point)
    except Exception:
        point = None
    if point and point > 0.0:
        df["spread"] = df["spread"] * point
        logging.info("Converted MT5 spread from points to price for %s (point=%s).", symbol, point)
    df = df[["bar_dt", "open", "high", "low", "close", "volume", "spread"]]
    df = _normalize_bar_dt(df)
    M1_CACHE_MT5[symbol] = df.copy()
    return df


def _find_csv_candidates(symbol: str, folder: Path) -> list[Path]:
    candidates = []
    for path in folder.rglob("*.csv"):
        name = path.name.upper()
        if symbol.upper() in name and ("M1" in name or "1MIN" in name or "1_MIN" in name or "1-MIN" in name):
            candidates.append(path)
    return candidates


def load_m1_from_csv(symbol: str, folder: Path = M1_DATA_DIR) -> pd.DataFrame:
    if symbol in M1_CACHE_CSV:
        return M1_CACHE_CSV[symbol].copy()

    candidates = _find_csv_candidates(symbol, folder)
    if not candidates:
        return pd.DataFrame(columns=["bar_dt", "open", "high", "low", "close", "volume", "spread"])

    best_df = pd.DataFrame()
    best_rows = 0
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        dt_col = None
        for col in ("bar_dt", "datetime", "time", "date"):
            if col in df.columns:
                dt_col = col
                break
        if dt_col is None:
            continue
        df = df.rename(columns={dt_col: "bar_dt"})
        df["bar_dt"] = _to_utc_series(df["bar_dt"])
        df["open"] = df.get("open", df.get("Open", np.nan))
        df["high"] = df.get("high", df.get("High", np.nan))
        df["low"] = df.get("low", df.get("Low", np.nan))
        df["close"] = df.get("close", df.get("Close", np.nan))
        df["volume"] = df.get("volume", df.get("tick_volume", df.get("real_volume", 0.0)))
        df["spread"] = df.get("spread", df.get("Spread", 0.0))
        df = df[["bar_dt", "open", "high", "low", "close", "volume", "spread"]]
        df = _normalize_bar_dt(df)
        if len(df) > best_rows:
            best_rows = len(df)
            best_df = df

    if best_df.empty:
        return pd.DataFrame(columns=["bar_dt", "open", "high", "low", "close", "volume", "spread"])

    M1_CACHE_CSV[symbol] = best_df.copy()
    return best_df


def compare_span_and_choose(symbol: str) -> pd.DataFrame:
    mt5_df = load_m1_from_mt5(symbol)
    csv_df = load_m1_from_csv(symbol)

    def span_info(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
        if df.empty:
            return (pd.Timestamp.max.tz_localize("UTC"), pd.Timestamp.min.tz_localize("UTC"), 0)
        return (df["bar_dt"].iloc[0], df["bar_dt"].iloc[-1], len(df))

    mt5_start, _mt5_end, mt5_rows = span_info(mt5_df)
    csv_start, _csv_end, csv_rows = span_info(csv_df)

    use_csv = False
    if csv_rows > mt5_rows:
        use_csv = True
    elif csv_rows > 0 and csv_start < mt5_start:
        use_csv = True

    chosen = csv_df if use_csv else mt5_df
    source = "CSV" if use_csv else "MT5"
    if chosen.empty:
        logging.info("M1 bars for %s missing from both MT5 and CSV.", symbol)
    else:
        logging.info(
            "M1 source %s for %s: %s -> %s rows=%s",
            source,
            symbol,
            chosen["bar_dt"].iloc[0],
            chosen["bar_dt"].iloc[-1],
            len(chosen),
        )
    return chosen


def load_m1_bars(symbol: str) -> pd.DataFrame:
    return compare_span_and_choose(symbol)


def _select_usd_index_symbol() -> str:
    candidates = [USD_INDEX_SYMBOL] + [s for s in USD_INDEX_ALIASES if s != USD_INDEX_SYMBOL]
    for sym in candidates:
        df = load_m1_from_csv(sym)
        if not df.empty:
            return sym
    for sym in candidates:
        df = load_m1_from_mt5(sym)
        if not df.empty:
            return sym
    msg = (
        "No USD index proxy data found. Configure USD_INDEX_SYMBOL or provide CSVs. "
        "Tried symbols: " + ", ".join(candidates)
    )
    raise FileNotFoundError(msg)

@dataclass
class LevelState:
    centers: np.ndarray
    last_build_ts: Optional[pd.Timestamp]
    touches: list[tuple[pd.Timestamp, int]]
    bounces: list[tuple[pd.Timestamp, int]]
    pending: list[dict]


@dataclass
class ChannelState:
    last_fit_idx: int
    lower: tuple[float, float]
    mid: tuple[float, float]
    upper: tuple[float, float]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TCNFrontend(nn.Module):
    def __init__(self, in_features: int, channels: int, kernel: int, layers: int, dropout: float) -> None:
        super().__init__()
        net: list[nn.Module] = []
        input_channels = in_features
        for _ in range(layers):
            net.append(nn.Conv1d(input_channels, channels, kernel, padding=kernel - 1))
            net.append(nn.ReLU())
            net.append(nn.Dropout(dropout))
            input_channels = channels
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.net(x)
        return out.transpose(1, 2)


class ContextTCNTransformer(nn.Module):
    def __init__(
        self,
        in_features: int,
        seq_len: int,
        embed_dim: int,
        layers: int,
        nhead: int,
        dropout: float,
        tcn_channels: int,
        tcn_kernel: int,
        tcn_layers: int,
    ) -> None:
        super().__init__()
        self.tcn = TCNFrontend(in_features, tcn_channels, tcn_kernel, tcn_layers, dropout)
        self.proj = nn.Linear(tcn_channels, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        emb = self.tcn(x)
        if emb.size(1) != seq_len:
            emb = emb[:, -seq_len:, :]
        emb = self.proj(emb)
        emb = self.pos_enc(emb)
        if torch.cuda.is_available() and hasattr(torch.backends.cuda, "sdp_kernel"):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            ):
                encoded = self.encoder(emb)
        else:
            encoded = self.encoder(emb)
        return encoded.mean(dim=1)


class TrendGateModule(nn.Module):
    def __init__(self, vol_idx: Optional[int]) -> None:
        super().__init__()
        self.vol_idx = vol_idx
        self.tau_slope = nn.Parameter(torch.tensor(TAU_SLOPE_INIT, dtype=torch.float32))
        self.tau_push = nn.Parameter(torch.tensor(TAU_PUSH_INIT, dtype=torch.float32))
        self.tau_pull = nn.Parameter(torch.tensor(TAU_PULL_INIT, dtype=torch.float32))
        self.tau_ratio = nn.Parameter(torch.tensor(TAU_RATIO_INIT, dtype=torch.float32))
        self.tau_mode = nn.Parameter(torch.tensor(TAU_MODE_INIT, dtype=torch.float32))

    def _slope(self, price: torch.Tensor) -> torch.Tensor:
        t = torch.arange(price.size(1), device=price.device, dtype=price.dtype)
        t = t - t.mean()
        x = price - price.mean(dim=1, keepdim=True)
        cov = (x * t).sum(dim=1)
        var = (t * t).sum() + TREND_EPS
        return cov / var

    def _pool_long(self, seq: torch.Tensor) -> torch.Tensor:
        if TREND_LONG_POOL_K <= 1:
            return seq
        squeezed = False
        if seq.dim() == 2:
            seq = seq.unsqueeze(-1)
            squeezed = True
        steps = seq.size(1) // TREND_LONG_POOL_K
        if steps <= 0:
            return seq.squeeze(-1) if squeezed else seq
        seq = seq[:, -steps * TREND_LONG_POOL_K :, :]
        seq = seq.view(seq.size(0), steps, TREND_LONG_POOL_K, seq.size(2))
        if TREND_LONG_POOL_MODE == "last":
            pooled = seq[:, :, -1, :]
        else:
            pooled = seq.mean(dim=2)
        return pooled.squeeze(-1) if squeezed else pooled

    # TREND-GATE FIX: trend gate uses raw close series (not model features)
    def forward(
        self,
        close_short: torch.Tensor,
        close_long: torch.Tensor,
        vol_short: Optional[torch.Tensor] = None,
        vol_long: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        close_long = self._pool_long(close_long)
        slope_short = self._slope(close_short)
        slope_long = self._slope(close_long)
        close_last = close_short[:, -1]
        if vol_short is not None:
            vol_last = vol_short[:, -1]
            atr_px = torch.abs(vol_last * close_last) + TREND_EPS
        else:
            atr_px = torch.abs(close_short[:, -1] - close_short[:, -2]) + TREND_EPS

        slope_combined = (slope_short + slope_long) * 0.5
        slope_norm = slope_combined / atr_px

        w = min(TREND_SHORT_PUSH_WINDOW, close_short.size(1) - 1)
        delta = close_short[:, 1:] - close_short[:, :-1]
        delta_win = delta[:, -w:]
        push_up_atr = torch.max(torch.relu(delta_win), dim=1).values / atr_px
        push_dn_atr = torch.max(torch.relu(-delta_win), dim=1).values / atr_px

        price_win = close_short[:, -w:]
        peak = torch.cummax(price_win, dim=1).values
        trough = torch.cummin(price_win, dim=1).values
        pullback_dn_atr = torch.max((peak - price_win), dim=1).values / atr_px
        pullback_up_atr = torch.max((price_win - trough), dim=1).values / atr_px

        ratio_up = push_up_atr / (pullback_dn_atr + TREND_EPS)
        ratio_dn = push_dn_atr / (pullback_up_atr + TREND_EPS)

        g_slope_up = torch.sigmoid((slope_norm - self.tau_slope) / TEMP_SLOPE)
        g_slope_dn = torch.sigmoid((-slope_norm - self.tau_slope) / TEMP_SLOPE)
        g_push_up = torch.sigmoid((push_up_atr - self.tau_push) / TEMP_PUSH)
        g_push_dn = torch.sigmoid((push_dn_atr - self.tau_push) / TEMP_PUSH)
        g_pull_dn_small = torch.sigmoid((self.tau_pull - pullback_dn_atr) / TEMP_PULL)
        g_pull_up_small = torch.sigmoid((self.tau_pull - pullback_up_atr) / TEMP_PULL)
        g_ratio_up = torch.sigmoid((ratio_up - self.tau_ratio) / TEMP_RATIO)
        g_ratio_dn = torch.sigmoid((ratio_dn - self.tau_ratio) / TEMP_RATIO)

        score_up = g_slope_up * g_push_up * g_pull_dn_small * g_ratio_up
        score_dn = g_slope_dn * g_push_dn * g_pull_up_small * g_ratio_dn
        score = torch.maximum(score_up, score_dn)

        p_trend = torch.sigmoid((score - self.tau_mode) / TEMP_MODE)
        p_trend_up = torch.sigmoid((score_up - score_dn) / TEMP_DIR)
        return p_trend, p_trend_up


class EntryCombiner(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TransitionHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if TRANSITION_HEAD_HIDDEN and TRANSITION_HEAD_HIDDEN > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, TRANSITION_HEAD_HIDDEN),
                nn.ReLU(),
                nn.Dropout(TRANSITION_HEAD_DROPOUT),
                nn.Linear(TRANSITION_HEAD_HIDDEN, 1),
            )
        else:
            self.net = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TrendMRModel(nn.Module):
    def __init__(self, in_features: int, slow_in_features: int) -> None:
        super().__init__()
        self.use_slow = bool(USE_SLOW_STREAM)
        self.short_ctx = ContextTCNTransformer(
            in_features, SHORT_LEN, EMBED_DIM, LAYERS, NHEAD, DROPOUT, TCN_CHANNELS, TCN_KERNEL, TCN_LAYERS
        )
        self.mid_ctx = ContextTCNTransformer(
            in_features, MID_LEN, EMBED_DIM, LAYERS, NHEAD, DROPOUT, TCN_CHANNELS, TCN_KERNEL, TCN_LAYERS
        )
        self.long_ctx = ContextTCNTransformer(
            in_features, LONG_LEN, EMBED_DIM, LAYERS, NHEAD, DROPOUT, TCN_CHANNELS, TCN_KERNEL, TCN_LAYERS
        )
        if self.use_slow:
            self.slow_ctx = ContextTCNTransformer(
                slow_in_features, SLOW_LEN, EMBED_DIM, LAYERS, NHEAD, DROPOUT, TCN_CHANNELS, TCN_KERNEL, TCN_LAYERS
            )
        else:
            self.slow_ctx = None
        self.shared = nn.Sequential(
            nn.Linear(EMBED_DIM * 4, EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        self.trade_head = nn.Linear(EMBED_DIM, 1)
        self.dir_head = nn.Linear(EMBED_DIM, 1)
        self.bounce_head = nn.Linear(EMBED_DIM, 1)
        if TTP_HEAD_HIDDEN and TTP_HEAD_HIDDEN > 0:
            self.ttp_head = nn.Sequential(
                nn.Linear(EMBED_DIM, TTP_HEAD_HIDDEN),
                nn.ReLU(),
                nn.Dropout(TTP_HEAD_DROPOUT),
                nn.Linear(TTP_HEAD_HIDDEN, 1),
            )
        else:
            self.ttp_head = nn.Linear(EMBED_DIM, 1)
        if EDGE_HEAD_HIDDEN and EDGE_HEAD_HIDDEN > 0:
            self.edge_head = nn.Sequential(
                nn.Linear(EMBED_DIM, EDGE_HEAD_HIDDEN),
                nn.ReLU(),
                nn.Dropout(EDGE_HEAD_DROPOUT),
                nn.Linear(EDGE_HEAD_HIDDEN, 1),
            )
        else:
            self.edge_head = nn.Linear(EMBED_DIM, 1)
        self.qdist_head = nn.Linear(EMBED_DIM, len(QDIST_QUANTILES))
        self.trend_gate = TrendGateModule(vol_idx=TREND_VOL_COL_IDX)
        self.entry_combiner = EntryCombiner(_combiner_input_dim(), COMBINER_HIDDEN, COMBINER_DROPOUT)
        if USE_TRANSITION_HEAD:
            self.transition_head = TransitionHead(_transition_input_dim())
        else:
            self.transition_head = None
        self.dist_exit_head = nn.Linear(EMBED_DIM, len(DIST_EXIT_HORIZONS) * len(DIST_EXIT_QUANTILES))
        self.ksl_head = nn.Linear(EMBED_DIM, 1)
        self.ktp_head = nn.Linear(EMBED_DIM, 1)
        self.break_head = nn.Linear(EMBED_DIM, 1)

    def forward(
        self,
        short_seq: torch.Tensor,
        mid_seq: torch.Tensor,
        long_seq: torch.Tensor,
        slow_seq: torch.Tensor,
        trend_short_seq: torch.Tensor,
        trend_long_seq: torch.Tensor,
        close_short: torch.Tensor,
        close_long: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        short_repr = self.short_ctx(short_seq)
        mid_repr = self.mid_ctx(mid_seq)
        long_repr = self.long_ctx(long_seq)
        if self.slow_ctx is None:
            slow_repr = torch.zeros_like(short_repr)
        else:
            slow_repr = self.slow_ctx(slow_seq)
        combined = torch.cat([short_repr, mid_repr, long_repr, slow_repr], dim=1)
        shared = self.shared(combined)
        vol_short = trend_short_seq[:, :, TREND_VOL_COL_IDX] if TREND_VOL_COL_IDX is not None else None
        # TREND-GATE FIX: trend gate uses raw close series (not model features)
        p_trend, p_trend_up = self.trend_gate(close_short, close_long, vol_short=vol_short)
        transition_logit = torch.zeros_like(p_trend)
        if USE_TRANSITION_HEAD and self.transition_head is not None:
            price_seq = close_short.unsqueeze(-1)
            slopes = _compute_slopes_multi_torch(price_seq, SLOPE_WINDOWS)
            if SLOPE_USE_ATR_NORM:
                if TREND_VOL_COL_IDX is not None:
                    atr_px = trend_short_seq[:, -1, TREND_VOL_COL_IDX]
                else:
                    atr_px = torch.abs(price_seq[:, -1, 0] - price_seq[:, -2, 0])
                atr_px = atr_px.clamp_min(SLOPE_ATR_EPS)
                slopes_norm = slopes / atr_px.unsqueeze(-1)
            else:
                slopes_norm = slopes
            slopes_norm = torch.clamp(slopes_norm, -DISAGREE_CLIP, DISAGREE_CLIP)
            disagree_feats = _compute_disagree_features_torch(slopes_norm)
            inputs = []
            if TRANSITION_INCLUDE_SHARED:
                inputs.append(shared)
            if TRANSITION_INCLUDE_SLOPES:
                inputs.append(slopes_norm)
            if TRANSITION_INCLUDE_DISAGREE:
                inputs.append(disagree_feats)
            trans_in = torch.cat(inputs, dim=-1) if inputs else shared
            transition_logit = self.transition_head(trans_in)
        dist_raw = self.dist_exit_head(shared)
        dist_raw = dist_raw.view(-1, len(DIST_EXIT_HORIZONS), len(DIST_EXIT_QUANTILES))
        dist_sorted, _ = torch.sort(dist_raw, dim=-1)
        dist_sorted = torch.clamp(dist_sorted, DIST_EXIT_CLAMP_MIN, DIST_EXIT_CLAMP_MAX)
        edge_pred = self.edge_head(shared).squeeze(-1)
        qdist_pred = self.qdist_head(shared)
        qdist_pred = torch.clamp(qdist_pred, -QDIST_CLIP_RET_ATR, QDIST_CLIP_RET_ATR)
        return (
            self.trade_head(shared).squeeze(-1),
            self.dir_head(shared).squeeze(-1),
            self.bounce_head(shared).squeeze(-1),
            self.ttp_head(shared).squeeze(-1),
            edge_pred,
            qdist_pred,
            p_trend,
            F.softplus(self.ksl_head(shared)).squeeze(-1),
            F.softplus(self.ktp_head(shared)).squeeze(-1),
            dist_sorted,
            torch.sigmoid(self.break_head(shared)).squeeze(-1),
            p_trend_up,
            transition_logit,
        )


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        close_prices: np.ndarray,
        slow_features: Optional[np.ndarray],
        slow_map: Optional[np.ndarray],
        labels: dict[str, np.ndarray],
        bar_ts: np.ndarray,
        use_slow: bool,
        feature_dim: int,
    ) -> None:
        self.features = features
        if close_prices is None or close_prices.size == 0:
            raise ValueError("Close prices are required for trend gate windows.")
        self.close_prices = close_prices
        self.slow_features = slow_features
        self.slow_map = slow_map
        self.labels = labels
        self.bar_ts = bar_ts
        self.use_slow = use_slow
        self.slow_dim = feature_dim if slow_features is None else slow_features.shape[1]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        short_seq = _build_sequence(self.features, idx, SHORT_LEN)
        mid_seq = _build_sequence(self.features, idx, MID_LEN)
        long_seq = _build_sequence(self.features, idx, LONG_LEN)
        trend_short_seq = _build_sequence(self.features, idx, TREND_SHORT_LOOKBACK_BARS)
        trend_long_seq = _build_sequence(self.features, idx, TREND_LONG_LOOKBACK_BARS)
        close_short_seq = _build_close_sequence(self.close_prices, idx, TREND_SHORT_LOOKBACK_BARS)
        close_long_seq = _build_close_sequence(self.close_prices, idx, TREND_LONG_LOOKBACK_BARS)
        if self.use_slow and self.slow_features is not None and self.slow_map is not None:
            slow_idx = int(self.slow_map[idx]) if len(self.slow_map) else 0
            slow_seq = _build_sequence(self.slow_features, slow_idx, SLOW_LEN)
        else:
            slow_seq = np.zeros((SLOW_LEN, self.slow_dim), dtype=np.float32)
        return {
            "short": torch.from_numpy(short_seq),
            "mid": torch.from_numpy(mid_seq),
            "long": torch.from_numpy(long_seq),
            "trend_short": torch.from_numpy(trend_short_seq),
            "trend_long": torch.from_numpy(trend_long_seq),
            "trend_close_short": torch.from_numpy(close_short_seq),
            "trend_close_long": torch.from_numpy(close_long_seq),
            "slow": torch.from_numpy(slow_seq),
            "y_trade": torch.tensor(self.labels["y_trade"][idx], dtype=torch.float32),
            "y_dir": torch.tensor(self.labels["y_dir"][idx], dtype=torch.float32),
            "y_bounce": torch.tensor(self.labels["y_bounce"][idx], dtype=torch.float32),
            "y_trend": torch.tensor(self.labels["y_trend"][idx], dtype=torch.float32),
            "y_trend_up": torch.tensor(self.labels["y_trend_up"][idx], dtype=torch.float32),
            "y_trend_up_soft": torch.tensor(self.labels["y_trend_up_soft"][idx], dtype=torch.float32),
            "y_transition": torch.tensor(self.labels["y_transition"][idx], dtype=torch.float32),
            "mask_dir": torch.tensor(self.labels["mask_dir"][idx], dtype=torch.float32),
            "mask_bounce": torch.tensor(self.labels["mask_bounce"][idx], dtype=torch.float32),
            "mask_trend": torch.tensor(self.labels["mask_trend"][idx], dtype=torch.float32),
            "mask_transition": torch.tensor(self.labels["mask_transition"][idx], dtype=torch.float32),
            "y_ksl": torch.tensor(self.labels["y_ksl"][idx], dtype=torch.float32),
            "mask_ksl": torch.tensor(self.labels["mask_ksl"][idx], dtype=torch.float32),
            "y_ktp": torch.tensor(self.labels["y_ktp"][idx], dtype=torch.float32),
            "mask_ktp": torch.tensor(self.labels["mask_ktp"][idx], dtype=torch.float32),
            "y_ttp_quick": torch.tensor(self.labels["y_ttp_quick"][idx], dtype=torch.float32),
            "mask_ttp": torch.tensor(self.labels["mask_ttp"][idx], dtype=torch.float32),
            "y_edge": torch.tensor(self.labels["y_edge"][idx], dtype=torch.float32),
            "mask_edge": torch.tensor(self.labels["mask_edge"][idx], dtype=torch.float32),
            "y_qdist": torch.tensor(self.labels["y_qdist"][idx], dtype=torch.float32),
            "mask_qdist": torch.tensor(self.labels["mask_qdist"][idx], dtype=torch.float32),
            "y_dist_targets": torch.tensor(self.labels["y_dist_targets"][idx], dtype=torch.float32),
            "mask_dist": torch.tensor(self.labels["mask_dist"][idx], dtype=torch.float32),
            "y_break": torch.tensor(self.labels["y_break"][idx], dtype=torch.float32),
            "mask_break": torch.tensor(self.labels["mask_break"][idx], dtype=torch.float32),
            "excess_up_atr": torch.tensor(self.labels["excess_up_atr"][idx], dtype=torch.float32),
            "excess_dn_atr": torch.tensor(self.labels["excess_dn_atr"][idx], dtype=torch.float32),
            "trend_strength": torch.tensor(self.labels["trend_strength"][idx], dtype=torch.float32),
            "trend_strength_norm": torch.tensor(self.labels["trend_strength_norm"][idx], dtype=torch.float32),
            "trend_adv": torch.tensor(self.labels["trend_adv"][idx], dtype=torch.float32),
            "vol_30m": torch.tensor(self.labels["vol_30m"][idx], dtype=torch.float32),
            "channel_slope": torch.tensor(self.labels["channel_slope"][idx], dtype=torch.float32),
            "xau_spread": torch.tensor(self.labels["xau_spread"][idx], dtype=torch.float32),
            "bar_ts": self.bar_ts[idx],
        }


def _build_sequence(matrix: np.ndarray, idx: int, seq_len: int) -> np.ndarray:
    start = idx - seq_len + 1
    if start < 0:
        pad = np.zeros((-start, matrix.shape[1]), dtype=np.float32)
        seq = np.concatenate([pad, matrix[0 : idx + 1].astype(np.float32)], axis=0)
    else:
        seq = matrix[start : idx + 1].astype(np.float32)
    if seq.shape[0] != seq_len:
        pad = np.zeros((seq_len - seq.shape[0], matrix.shape[1]), dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)
    return seq


def _build_close_sequence(closes: np.ndarray, idx: int, seq_len: int) -> np.ndarray:
    if closes.size == 0:
        raise ValueError("Close series is empty; cannot build trend gate windows.")
    start = idx - seq_len + 1
    if start < 0:
        pad_val = float(closes[0]) if np.isfinite(closes[0]) else 0.0
        pad = np.full((-start,), pad_val, dtype=np.float32)
        seq = np.concatenate([pad, closes[0 : idx + 1].astype(np.float32)], axis=0)
    else:
        seq = closes[start : idx + 1].astype(np.float32)
    if seq.shape[0] != seq_len:
        pad_val = float(seq[0]) if seq.size else 0.0
        pad = np.full((seq_len - seq.shape[0],), pad_val, dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)
    if not np.isfinite(seq).all():
        raise ValueError("Non-finite close values found in trend gate window.")
    return seq

def _session_flags(ts: pd.Timestamp) -> tuple[int, int, int, int, int, float, float]:
    hour = ts.hour
    minute = ts.minute
    asia = 1 if 0 <= hour < 7 else 0
    london = 1 if 7 <= hour < 13 else 0
    ny = 1 if 13 <= hour < 20 else 0
    london_open = 1 if hour == 7 and minute < SESSION_OPEN_WINDOW_MIN else 0
    ny_open = 1 if hour == 13 and minute < SESSION_OPEN_WINDOW_MIN else 0
    minute_of_day = hour * 60 + minute
    ang = 2 * math.pi * (minute_of_day / 1440.0)
    return asia, london, ny, london_open, ny_open, math.sin(ang), math.cos(ang)


def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var.replace(0.0, np.nan)


def _sym_info_safe(symbol: str) -> dict:
    info = mt5.symbol_info(symbol) if mt5 is not None else None
    if info is None:
        return {}
    return {
        "digits": getattr(info, "digits", None),
        "point": getattr(info, "point", None),
        "trade_stops_level": getattr(info, "trade_stops_level", None),
        "trade_freeze_level": getattr(info, "trade_freeze_level", None),
        "volume_min": getattr(info, "volume_min", None),
        "volume_max": getattr(info, "volume_max", None),
        "volume_step": getattr(info, "volume_step", None),
        "trade_mode": getattr(info, "trade_mode", None),
    }


def _get_tick(symbol: str):
    return mt5.symbol_info_tick(symbol) if mt5 is not None else None


def _get_symbol_info(symbol: str):
    return mt5.symbol_info(symbol) if mt5 is not None else None


def _round_price(price: float, digits: int) -> float:
    return round(float(price), int(digits))


def _timeframe_to_minutes(timeframe: int) -> int:
    if mt5 is None:
        return 1
    mapping = {
        mt5.TIMEFRAME_M1: 1,
        mt5.TIMEFRAME_M2: 2,
        mt5.TIMEFRAME_M3: 3,
        mt5.TIMEFRAME_M4: 4,
        mt5.TIMEFRAME_M5: 5,
        mt5.TIMEFRAME_M10: 10,
        mt5.TIMEFRAME_M15: 15,
        mt5.TIMEFRAME_M30: 30,
        mt5.TIMEFRAME_H1: 60,
        mt5.TIMEFRAME_H4: 240,
    }
    return int(mapping.get(timeframe, 1))


def _compute_er_and_signed_er(closes: np.ndarray) -> tuple[float, float]:
    net = float(closes[-1] - closes[0])
    net_abs = abs(net)
    path = float(np.abs(np.diff(closes)).sum())
    er = 0.0 if path <= 0 else net_abs / path
    signed_er = 0.0 if er == 0 else (1.0 if net > 0 else -1.0) * er
    return er, signed_er


def _positions_by_magic(symbol: str, magic: int) -> list:
    positions = mt5.positions_get(symbol=symbol) if mt5 is not None else None
    if positions is None:
        return []
    out = []
    for p in positions:
        if hasattr(p, "magic") and int(p.magic) == int(magic):
            out.append(p)
    return out


def _min_stop_distance(symbol_info) -> float:
    stops = int(getattr(symbol_info, "trade_stops_level", 0) or 0)
    freeze = int(getattr(symbol_info, "trade_freeze_level", 0) or 0)
    point = float(getattr(symbol_info, "point", 0.0) or 0.0)
    return max(stops, freeze) * point


def _validate_sltp_for_position(
    pos_type: int,
    bid: float,
    ask: float,
    sl: float,
    tp: float,
    min_dist: float,
) -> tuple[bool, bool, str, str]:
    sl_ok, tp_ok = True, True
    sl_reason, tp_reason = "", ""

    if pos_type == mt5.POSITION_TYPE_BUY:
        if sl and not (sl < bid):
            sl_ok = False
            sl_reason = f"BUY SL must be < bid (sl={sl} bid={bid})"
        if tp and not (tp > ask):
            tp_ok = False
            tp_reason = f"BUY TP must be > ask (tp={tp} ask={ask})"
        if sl and (bid - sl) < min_dist:
            sl_ok = False
            sl_reason = f"BUY SL too close (bid-sl={bid - sl} < min_dist={min_dist})"
        if tp and (tp - ask) < min_dist:
            tp_ok = False
            tp_reason = f"BUY TP too close (tp-ask={tp - ask} < min_dist={min_dist})"
        if sl and tp and not (sl < tp):
            sl_ok = False
            tp_ok = False
            sl_reason = tp_reason = f"BUY requires sl < tp (sl={sl} tp={tp})"
    else:
        if sl and not (sl > ask):
            sl_ok = False
            sl_reason = f"SELL SL must be > ask (sl={sl} ask={ask})"
        if tp and not (tp < bid):
            tp_ok = False
            tp_reason = f"SELL TP must be < bid (tp={tp} bid={bid})"
        if sl and (sl - ask) < min_dist:
            sl_ok = False
            sl_reason = f"SELL SL too close (sl-ask={sl - ask} < min_dist={min_dist})"
        if tp and (bid - tp) < min_dist:
            tp_ok = False
            tp_reason = f"SELL TP too close (bid-tp={bid - tp} < min_dist={min_dist})"
        if sl and tp and not (tp < sl):
            sl_ok = False
            tp_ok = False
            sl_reason = tp_reason = f"SELL requires tp < sl (tp={tp} sl={sl})"
    return sl_ok, tp_ok, sl_reason, tp_reason


def _maybe_send_sltp_modify(ticket: int, symbol: str, sl: float, tp: float, magic: int, comment: str):
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": int(ticket),
        "symbol": symbol,
        "sl": float(sl) if sl else 0.0,
        "tp": float(tp) if tp else 0.0,
        "magic": int(magic),
        "comment": comment,
    }
    res = mt5.order_send(req)
    return req, res


def _round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(x / step) * step


def _validate_sltp(
    side: str,
    entry: float,
    bid: float,
    ask: float,
    sl: Optional[float],
    tp: Optional[float],
    stops_level_points: Optional[float],
    freeze_level_points: Optional[float],
    point: Optional[float],
) -> tuple[bool, bool, str, str]:
    min_points = max(stops_level_points or 0.0, freeze_level_points or 0.0)
    min_dist = min_points * (point or 0.0)
    sl_ok, tp_ok = True, True
    sl_reason, tp_reason = "", ""

    if side == "BUY":
        if sl and not (sl < bid):
            sl_ok = False
            sl_reason = f"BUY SL must be < bid (sl={sl} bid={bid})"
        if tp and not (tp > ask):
            tp_ok = False
            tp_reason = f"BUY TP must be > ask (tp={tp} ask={ask})"
        if sl and (bid - sl) < min_dist:
            sl_ok = False
            sl_reason = f"BUY SL too close (bid-sl={bid - sl} < min_dist={min_dist})"
        if tp and (tp - ask) < min_dist:
            tp_ok = False
            tp_reason = f"BUY TP too close (tp-ask={tp - ask} < min_dist={min_dist})"
    else:
        if sl and not (sl > ask):
            sl_ok = False
            sl_reason = f"SELL SL must be > ask (sl={sl} ask={ask})"
        if tp and not (tp < bid):
            tp_ok = False
            tp_reason = f"SELL TP must be < bid (tp={tp} bid={bid})"
        if sl and (sl - ask) < min_dist:
            sl_ok = False
            sl_reason = f"SELL SL too close (sl-ask={sl - ask} < min_dist={min_dist})"
        if tp and (bid - tp) < min_dist:
            tp_ok = False
            tp_reason = f"SELL TP too close (bid-tp={bid - tp} < min_dist={min_dist})"

    if sl and tp:
        if side == "BUY" and not (sl < tp):
            sl_ok = False
            tp_ok = False
            sl_reason = tp_reason = f"BUY requires sl < tp (sl={sl} tp={tp})"
        if side == "SELL" and not (tp < sl):
            sl_ok = False
            tp_ok = False
            sl_reason = tp_reason = f"SELL requires tp < sl (tp={tp} sl={sl})"

    return sl_ok, tp_ok, sl_reason, tp_reason


def _safe_corrcoef(x: np.ndarray, y: np.ndarray, tag: str) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        if tag not in _CORR_WARNED:
            logging.warning("corrcoef skipped (%s): non-finite values in window.", tag)
            _CORR_WARNED.add(tag)
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        if tag not in _CORR_WARNED:
            logging.warning("corrcoef skipped (%s): zero stddev in window.", tag)
            _CORR_WARNED.add(tag)
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _fit_channel_fast(
    window: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Fast channel fit:
    - OLS midline y = intercept + slope*x
    - Residual quantiles for lower/upper offsets
    Returns (lower_line, mid_line, upper_line) as (intercept, slope).
    """
    x = np.arange(len(window), dtype=np.float64)
    slope, intercept = np.polyfit(x, window.astype(np.float64), 1)
    mid = intercept + slope * x
    resid = window - mid
    lower_off = float(np.quantile(resid, CHANNEL_Q_LOWER))
    upper_off = float(np.quantile(resid, CHANNEL_Q_UPPER))
    return (intercept + lower_off, slope), (intercept, slope), (intercept + upper_off, slope)


def _fallback_channel(y: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return _fit_channel_fast(y)


def _update_channel_state(state: ChannelState, prices: np.ndarray, idx: int) -> ChannelState:
    if idx < CHANNEL_WINDOW_MINUTES:
        return state
    if state.last_fit_idx >= 0 and (idx - state.last_fit_idx) < CHANNEL_REFIT_MINUTES:
        return state
    window = prices[idx - CHANNEL_WINDOW_MINUTES + 1 : idx + 1]
    if state.last_fit_idx < 0:
        logging.info(
            "Channel fit: FAST OLS + residual quantiles (window=%s refit=%smin)",
            CHANNEL_WINDOW_MINUTES,
            CHANNEL_REFIT_MINUTES,
        )
    fit_start = time.perf_counter()
    logging.info("Channel fit: start (bars=%s idx=%s)", CHANNEL_WINDOW_MINUTES, idx)
    lower, mid, upper = _fit_channel_fast(window)
    fit_ms = (time.perf_counter() - fit_start) * 1000.0
    logging.info("Channel fit: done in %.1fms", fit_ms)
    return ChannelState(idx, lower, mid, upper)


def _channel_features(state: ChannelState, price: float) -> tuple[float, float, float, float, float, float, float]:
    if state.last_fit_idx < 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    idx = CHANNEL_WINDOW_MINUTES - 1
    lower = state.lower[0] + state.lower[1] * idx
    mid = state.mid[0] + state.mid[1] * idx
    upper = state.upper[0] + state.upper[1] * idx
    width = max(upper - lower, 1e-8)
    slope = state.mid[1]
    flatness = abs(slope) / (width + 1e-8)
    pos = (price - mid) / (0.5 * width + 1e-8)
    pos = float(np.clip(pos, -2.0, 2.0))
    break_flag = 1.0 if price > upper or price < lower else 0.0
    dist_upper = (upper - price) / (width + 1e-8)
    dist_lower = (price - lower) / (width + 1e-8)
    return slope, width, flatness, pos, break_flag, dist_upper, dist_lower


def _build_levels(prices: np.ndarray) -> np.ndarray:
    if KMeans is None:
        raise RuntimeError("sklearn not available for KMeans levels.")
    kmeans = KMeans(n_clusters=LEVEL_K, n_init=10, random_state=42)
    kmeans.fit(prices.reshape(-1, 1))
    return np.sort(kmeans.cluster_centers_.flatten())


def _level_features(
    state: LevelState,
    ts: pd.Timestamp,
    price: float,
    spread: float,
    atr_proxy: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    if state.centers.size == 0:
        return (0.0,) * 10
    levels = state.centers
    idx = int(np.argmin(np.abs(levels - price)))
    nearest = levels[idx]
    dist_nearest = abs(price - nearest)
    support_levels = levels[levels <= price]
    resistance_levels = levels[levels >= price]
    support = support_levels[-1] if support_levels.size else nearest
    resistance = resistance_levels[0] if resistance_levels.size else nearest
    dist_support = abs(price - support)
    dist_resistance = abs(resistance - price)
    atr = max(atr_proxy, 1e-8)
    dist_nearest_norm = dist_nearest / atr
    dist_support_norm = dist_support / atr
    dist_resistance_norm = dist_resistance / atr
    rank = idx / max(1, len(levels) - 1)

    touch_tol = max(spread, price * LEVEL_TOL_PCT)
    bounce_thresh = LEVEL_BOUNCE_MULT * touch_tol
    near_flag = 1.0 if dist_nearest <= touch_tol else 0.0

    state.touches.append((ts, idx))
    cutoff = ts - pd.Timedelta(hours=LEVEL_TOUCH_HOURS)
    state.touches = [(t, i) for (t, i) in state.touches if t >= cutoff]
    touch_count = sum(1 for (t, i) in state.touches if i == idx)

    if near_flag:
        state.pending.append({"level_idx": idx, "level": nearest, "time": ts, "resolved": False})
    for item in state.pending:
        if item["resolved"]:
            continue
        if abs(price - item["level"]) > bounce_thresh:
            item["resolved"] = True
            state.bounces.append((ts, item["level_idx"]))
    state.pending = [item for item in state.pending if item["time"] >= cutoff]
    state.bounces = [(t, i) for (t, i) in state.bounces if t >= cutoff]
    bounce_count = sum(1 for (t, i) in state.bounces if i == idx)

    return (
        dist_nearest,
        dist_support,
        dist_resistance,
        dist_nearest_norm,
        dist_support_norm,
        dist_resistance_norm,
        rank,
        near_flag,
        float(touch_count),
        float(bounce_count),
    )


def _check_and_fix_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> None:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    matrix = df[list(feature_cols)].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        bad_rows = np.where(~np.isfinite(matrix).all(axis=1))[0]
        sample_idx = bad_rows[:5].tolist()
        payload = {
            "bad_rows": sample_idx,
            "sample": df.iloc[sample_idx].to_dict(orient="records"),
        }
        path = OUTPUT_DIR / "feature_nan_rows.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
    df[list(feature_cols)] = df[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if not np.isfinite(df[list(feature_cols)].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite features remain after cleanup.")


def _build_feature_frame_from_sources(
    xau: pd.DataFrame,
    xag: pd.DataFrame,
    dxy: pd.DataFrame,
    *,
    log_progress: bool,
    live_fill: bool = False,
    progress_every: int = 200,
) -> tuple[pd.DataFrame, list[str]]:
    need_xag = USE_CROSS_ASSET or USE_CORR_BETA
    need_dxy = USE_CROSS_ASSET or USE_CORR_BETA or USE_XAUCORE
    if xau.empty or (need_xag and xag.empty) or (need_dxy and dxy.empty):
        raise ValueError("Missing required symbol data for features.")

    df = xau.rename(columns={"close": "xau_close", "spread": "xau_spread"})
    if need_xag:
        xag = xag.rename(columns={"close": "xag_close"})
        df = df.merge(xag[["bar_dt", "xag_close"]], on="bar_dt", how="left" if live_fill else "inner")
    if need_dxy:
        dxy = dxy.rename(columns={"close": "dxy_close"})
        df = df.merge(dxy[["bar_dt", "dxy_close"]], on="bar_dt", how="left" if live_fill else "inner")

    if live_fill:
        df = df.sort_values("bar_dt")
        if need_xag:
            df["xag_close"] = df["xag_close"].ffill()
        if need_dxy:
            df["dxy_close"] = df["dxy_close"].ffill()

    df["xau_close"] = pd.to_numeric(df["xau_close"], errors="coerce")
    if need_xag:
        df["xag_close"] = pd.to_numeric(df["xag_close"], errors="coerce")
    if need_dxy:
        df["dxy_close"] = pd.to_numeric(df["dxy_close"], errors="coerce")
    df["xau_spread"] = pd.to_numeric(df["xau_spread"], errors="coerce").fillna(0.0)
    df["xau_spread"] = df["xau_spread"] * SPREAD_POINTS_TO_PRICE

    df["ret_1m"] = df["xau_close"].pct_change()
    for w in RET_WINDOWS:
        if int(w) == 1:
            continue
        df[f"ret_{w}m"] = df["xau_close"].pct_change(int(w))
    start_timer("volatility")
    df["vol_30m"] = df["ret_1m"].rolling(VOL_WINDOW, min_periods=1).std()
    stop_timer("volatility")

    ret_xag = df["xag_close"].pct_change() if need_xag else pd.Series(0.0, index=df.index)
    ret_dxy = df["dxy_close"].pct_change() if need_dxy else pd.Series(0.0, index=df.index)

    if USE_CROSS_ASSET or USE_CORR_BETA or USE_XAUCORE:
        start_timer("correlations")
        for w in CORR_WINDOWS:
            if USE_CROSS_ASSET:
                df[f"corr_xau_xag_{w}"] = df["ret_1m"].rolling(w, min_periods=5).corr(ret_xag)
                df[f"corr_xau_dxy_{w}"] = df["ret_1m"].rolling(w, min_periods=5).corr(ret_dxy)
                df["sign_agree_xag"] = (np.sign(df["ret_1m"]) == np.sign(ret_xag)).astype(float)
                df["sign_agree_dxy"] = (np.sign(df["ret_1m"]) == -np.sign(ret_dxy)).astype(float)
            if USE_CORR_BETA:
                df[f"beta_xag_to_xau_{w}"] = _rolling_beta(ret_xag, df["ret_1m"], w)
                df[f"beta_xau_to_dxy_{w}"] = _rolling_beta(df["ret_1m"], ret_dxy, w)
            if USE_XAUCORE:
                beta = df[f"beta_xau_to_dxy_{w}"] if f"beta_xau_to_dxy_{w}" in df else np.nan
                df[f"xaucore_{w}"] = np.where(
                    np.isfinite(beta),
                    df["ret_1m"] + beta * ret_dxy,
                    df["ret_1m"] + ret_dxy,
                )
        stop_timer("correlations")

    if USE_SESSION_FLAGS:
        session_flags = df["bar_dt"].apply(_session_flags)
        df[
            ["session_asia", "session_london", "session_ny", "london_open", "ny_open", "minute_sin", "minute_cos"]
        ] = pd.DataFrame(session_flags.tolist(), index=df.index)

        df["session_id"] = np.where(df["session_asia"] == 1, 0, np.where(df["session_london"] == 1, 1, 2))
        vol_med_window = SESSION_MEDIAN_DAYS * 1440
        df["vol_session_median_5d"] = 0.0
        for sess in (0, 1, 2):
            mask = df["session_id"] == sess
            df.loc[mask, "vol_session_median_5d"] = (
                df.loc[mask, "vol_30m"].rolling(vol_med_window, min_periods=1).median().values
            )
        df["vol_session_ratio"] = df["vol_30m"] / (df["vol_session_median_5d"] + 1e-8)

    levels_state = LevelState(centers=np.array([]), last_build_ts=None, touches=[], bounces=[], pending=[])
    channel_state = ChannelState(last_fit_idx=-1, lower=(0.0, 0.0), mid=(0.0, 0.0), upper=(0.0, 0.0))

    level_cols = {
        "dist_nearest_level": [],
        "dist_support": [],
        "dist_resistance": [],
        "dist_nearest_norm": [],
        "dist_support_norm": [],
        "dist_resistance_norm": [],
        "level_rank": [],
        "near_level_flag": [],
        "touch_count_6h": [],
        "bounce_count_6h": [],
    }
    channel_cols = {
        "channel_slope": [],
        "channel_width": [],
        "channel_flatness": [],
        "pos_in_channel": [],
        "channel_break_flag": [],
        "dist_to_upper": [],
        "dist_to_lower": [],
    }
    channel_extra_cols = {
        "channel_upper": [],
        "channel_lower": [],
        "excess_up_atr": [],
        "excess_dn_atr": [],
    }

    if USE_LEVELS_KMEANS or USE_CHANNEL_QUANTREG:
        prices = df["xau_close"].to_numpy(dtype=np.float64)
        spreads = df["xau_spread"].to_numpy(dtype=np.float64)
        atr_proxy = (df["vol_30m"].fillna(0.0).to_numpy(dtype=np.float64) * prices)

        lookback_bars = LEVEL_LOOKBACK_DAYS * 1440
        rebuild_seconds = LEVEL_REBUILD_HOURS * 3600
        loop_start = time.perf_counter()

        for i, ts in enumerate(df["bar_dt"]):
            if log_progress and i % max(1, progress_every) == 0:
                elapsed = time.perf_counter() - loop_start
                logging.info("Feature loop progress: i=%s/%s elapsed=%.2fs", i, len(df), elapsed)
            if USE_LEVELS_KMEANS:
                if levels_state.last_build_ts is None or (ts - levels_state.last_build_ts).total_seconds() >= rebuild_seconds:
                    if i >= min(lookback_bars, len(prices) - 1):
                        window = prices[max(0, i - lookback_bars + 1) : i + 1]
                        logging.info(
                            "Levels rebuild: start (i=%s lookback=%s)",
                            i,
                            min(lookback_bars, len(prices) - 1),
                        )
                        rebuild_start = time.perf_counter()
                        start_timer("levels")
                        try:
                            levels_state.centers = _build_levels(window)
                            levels_state.last_build_ts = ts
                        except Exception as exc:
                            logging.info("Level build failed: %s", exc)
                        stop_timer("levels")
                        rebuild_ms = (time.perf_counter() - rebuild_start) * 1000.0
                        logging.info("Levels rebuild: done in %.1fms", rebuild_ms)

            if USE_CHANNEL_QUANTREG:
                start_timer("channel")
                channel_state = _update_channel_state(channel_state, prices, i)
                stop_timer("channel")

            if USE_LEVELS_KMEANS:
                lvl_start = time.perf_counter()
                start_timer("levels")
                lvl_feats = _level_features(
                    levels_state,
                    ts,
                    float(prices[i]),
                    float(spreads[i]),
                    float(atr_proxy[i]),
                )
                stop_timer("levels")
                if log_progress and i % max(1, progress_every) == 0:
                    lvl_ms = (time.perf_counter() - lvl_start) * 1000.0
                    logging.info("Levels features: i=%s done in %.1fms", i, lvl_ms)
                for key, val in zip(level_cols.keys(), lvl_feats):
                    level_cols[key].append(val)
            if USE_CHANNEL_QUANTREG:
                start_timer("channel")
                chan_feats = _channel_features(channel_state, float(prices[i]))
                stop_timer("channel")
                for key, val in zip(channel_cols.keys(), chan_feats):
                    channel_cols[key].append(val)
                idx_ref = CHANNEL_WINDOW_MINUTES - 1
                lower = channel_state.lower[0] + channel_state.lower[1] * idx_ref
                upper = channel_state.upper[0] + channel_state.upper[1] * idx_ref
                channel_extra_cols["channel_upper"].append(upper)
                channel_extra_cols["channel_lower"].append(lower)
                atr_px = float(atr_proxy[i]) if atr_proxy.size > i else 0.0
                if EXCEED_USE_CLOSE:
                    excess_up = max(0.0, (prices[i] - upper) / (atr_px + EXCEED_EPS))
                    excess_dn = max(0.0, (lower - prices[i]) / (atr_px + EXCEED_EPS))
                else:
                    excess_up = max(0.0, (df["high"].iloc[i] - upper) / (atr_px + EXCEED_EPS))
                    excess_dn = max(0.0, (lower - df["low"].iloc[i]) / (atr_px + EXCEED_EPS))
                channel_extra_cols["excess_up_atr"].append(excess_up)
                channel_extra_cols["excess_dn_atr"].append(excess_dn)

        if USE_LEVELS_KMEANS:
            for key, vals in level_cols.items():
                df[key] = vals
        if USE_CHANNEL_QUANTREG:
            for key, vals in channel_cols.items():
                df[key] = vals
            for key, vals in channel_extra_cols.items():
                df[key] = vals

    ret_cols = ["ret_1m"] + [f"ret_{w}m" for w in RET_WINDOWS if int(w) != 1]
    feature_cols = [*ret_cols, "vol_30m", "xau_spread"]
    if USE_CROSS_ASSET:
        for w in CORR_WINDOWS:
            feature_cols += [
                f"corr_xau_xag_{w}",
                f"corr_xau_dxy_{w}",
            ]
        feature_cols += ["sign_agree_xag", "sign_agree_dxy"]
    if USE_CORR_BETA:
        for w in CORR_WINDOWS:
            feature_cols += [
                f"beta_xag_to_xau_{w}",
                f"beta_xau_to_dxy_{w}",
            ]
    if USE_XAUCORE:
        for w in CORR_WINDOWS:
            feature_cols += [f"xaucore_{w}"]
    if USE_LEVELS_KMEANS:
        feature_cols += [
            "dist_nearest_level",
            "dist_support",
            "dist_resistance",
            "dist_nearest_norm",
            "dist_support_norm",
            "dist_resistance_norm",
            "level_rank",
            "touch_count_6h",
            "bounce_count_6h",
            "near_level_flag",
        ]
    if USE_CHANNEL_QUANTREG:
        feature_cols += [
            "channel_slope",
            "channel_width",
            "channel_flatness",
            "pos_in_channel",
            "channel_break_flag",
            "dist_to_upper",
            "dist_to_lower",
            "excess_up_atr",
            "excess_dn_atr",
        ]
    if USE_SESSION_FLAGS:
        feature_cols += [
            "session_asia",
            "session_london",
            "session_ny",
            "london_open",
            "ny_open",
            "minute_sin",
            "minute_cos",
            "vol_session_ratio",
        ]

    df = df.reset_index(drop=True)
    _check_and_fix_features(df, feature_cols)
    return df, feature_cols


def _build_feature_frame() -> tuple[pd.DataFrame, list[str]]:
    if CACHE_FEATURES and FEATURE_CACHE_PATH.exists() and FEATURE_CACHE_META.exists():
        try:
            meta = json.loads(FEATURE_CACHE_META.read_text(encoding="utf-8"))
            cached_cols = meta.get("feature_cols", [])
            cache_key = meta.get("cache_key")
            df = pd.read_parquet(FEATURE_CACHE_PATH)
            required_key = {
                "RET_WINDOWS": list(RET_WINDOWS),
                "CORR_WINDOWS": list(CORR_WINDOWS),
                "VOL_WINDOW": VOL_WINDOW,
                "toggles": {
                    "USE_CROSS_ASSET": USE_CROSS_ASSET,
                    "USE_CORR_BETA": USE_CORR_BETA,
                    "USE_XAUCORE": USE_XAUCORE,
                    "USE_LEVELS_KMEANS": USE_LEVELS_KMEANS,
                    "USE_CHANNEL_QUANTREG": USE_CHANNEL_QUANTREG,
                    "USE_SESSION_FLAGS": USE_SESSION_FLAGS,
                },
            }
            if cache_key == required_key and cached_cols and all(col in df.columns for col in cached_cols):
                logging.info("Loaded feature cache: %s rows=%s", FEATURE_CACHE_PATH, len(df))
                return df, list(cached_cols)
        except Exception as exc:
            logging.info("Feature cache load failed: %s", exc)

    usd_symbol = _select_usd_index_symbol()
    xau = load_m1_bars(TARGET_SYMBOL)
    need_xag = USE_CROSS_ASSET or USE_CORR_BETA
    need_dxy = USE_CROSS_ASSET or USE_CORR_BETA or USE_XAUCORE
    xag = load_m1_bars(XAG_SYMBOL) if need_xag else pd.DataFrame()
    dxy = load_m1_bars(usd_symbol) if need_dxy else pd.DataFrame()
    df, feature_cols = _build_feature_frame_from_sources(
        xau,
        xag,
        dxy,
        log_progress=True,
        live_fill=TRAIN_LIVE_FILL,
        progress_every=10,
    )
    if CACHE_FEATURES:
        try:
            FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(FEATURE_CACHE_PATH, index=False)
            cache_key = {
                "RET_WINDOWS": list(RET_WINDOWS),
                "CORR_WINDOWS": list(CORR_WINDOWS),
                "VOL_WINDOW": VOL_WINDOW,
                "toggles": {
                    "USE_CROSS_ASSET": USE_CROSS_ASSET,
                    "USE_CORR_BETA": USE_CORR_BETA,
                    "USE_XAUCORE": USE_XAUCORE,
                    "USE_LEVELS_KMEANS": USE_LEVELS_KMEANS,
                    "USE_CHANNEL_QUANTREG": USE_CHANNEL_QUANTREG,
                    "USE_SESSION_FLAGS": USE_SESSION_FLAGS,
                },
            }
            FEATURE_CACHE_META.write_text(
                json.dumps({"feature_cols": feature_cols, "cache_key": cache_key}, indent=2),
                encoding="utf-8",
            )
            logging.info("Saved feature cache: %s", FEATURE_CACHE_PATH)
        except Exception as exc:
            logging.info("Feature cache save failed: %s", exc)
    return df, feature_cols


def _build_slow_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    slow = df.set_index("bar_dt")[feature_cols].resample(f"{SLOW_FREQ_MINUTES}min").last().dropna().reset_index()
    slow_ts = slow["bar_dt"].to_numpy()
    fast_ts = df["bar_dt"].to_numpy()
    slow_map = np.searchsorted(slow_ts, fast_ts, side="right") - 1
    slow_map = np.clip(slow_map, 0, max(0, len(slow) - 1)).astype(np.int64)
    slow_features = slow[feature_cols].to_numpy(dtype=np.float32)
    return slow_features, slow_map


def _build_slow_sequence_live(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    slow = df.set_index("bar_dt")[feature_cols].resample(f"{SLOW_FREQ_MINUTES}min").last().dropna()
    if slow.empty:
        return np.zeros((SLOW_LEN, len(feature_cols)), dtype=np.float32)
    slow_vals = slow.to_numpy(dtype=np.float32)
    return _build_sequence(slow_vals, len(slow_vals) - 1, SLOW_LEN)


def _split_train_val(n_rows: int) -> tuple[int, int]:
    if TRAIN_END_IDX is not None and isinstance(TRAIN_END_IDX, int) and TRAIN_END_IDX > 0:
        train_end = min(int(TRAIN_END_IDX), n_rows)
    else:
        train_end = int(n_rows * TRAIN_SPLIT)
    return train_end, n_rows


def _normalize_features(
    df: pd.DataFrame, feature_cols: list[str], train_end: int
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    passthrough = {
        "sign_agree_xag",
        "sign_agree_dxy",
        "near_level_flag",
        "channel_break_flag",
        "session_asia",
        "session_london",
        "session_ny",
        "london_open",
        "ny_open",
    }
    norm_params: dict[str, tuple[float, float]] = {}
    matrix = df[feature_cols].to_numpy(dtype=np.float64)
    norm = matrix.copy()
    for i, col in enumerate(feature_cols):
        if col in passthrough:
            norm[:, i] = np.nan_to_num(norm[:, i], nan=0.0)
            continue
        train_slice = norm[:train_end, i]
        mean = float(np.nanmean(train_slice))
        std = float(np.nanstd(train_slice))
        if std <= 1e-8:
            std = 1.0
        norm[:, i] = (norm[:, i] - mean) / std
        norm_params[col] = (mean, std)
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return norm, norm_params


def _apply_norm_params(
    df: pd.DataFrame, feature_cols: list[str], norm_params: dict[str, tuple[float, float]]
) -> np.ndarray:
    passthrough = {
        "sign_agree_xag",
        "sign_agree_dxy",
        "near_level_flag",
        "channel_break_flag",
        "session_asia",
        "session_london",
        "session_ny",
        "london_open",
        "ny_open",
    }
    matrix = df[feature_cols].to_numpy(dtype=np.float64)
    norm = matrix.copy()
    for i, col in enumerate(feature_cols):
        if col in passthrough:
            norm[:, i] = np.nan_to_num(norm[:, i], nan=0.0)
            continue
        mean, std = norm_params.get(col, (0.0, 1.0))
        if std <= 1e-8:
            std = 1.0
        norm[:, i] = (norm[:, i] - mean) / std
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return norm


def _resolve_trend_indices(feature_cols: list[str]) -> None:
    global TREND_VOL_COL_IDX
    if TREND_VOL_COL_IDX is None:
        if TREND_VOL_COL_NAME in feature_cols:
            TREND_VOL_COL_IDX = feature_cols.index(TREND_VOL_COL_NAME)
        else:
            TREND_VOL_COL_IDX = None


def _compute_labels(df: pd.DataFrame, train_end: int) -> dict[str, np.ndarray]:
    prices = df["xau_close"].to_numpy(dtype=np.float64)
    spreads = df["xau_spread"].to_numpy(dtype=np.float64)
    horizon = max(1, int(HORIZON_MINUTES))
    future = np.roll(prices, -horizon)
    r_future = np.where(prices > 0.0, (future - prices) / np.maximum(prices, 1e-12), 0.0)
    r_future[-horizon:] = 0.0
    cost = np.where(prices > 0.0, spreads / np.maximum(prices, 1e-12), 0.0)
    train_abs = np.abs(r_future[:train_end])
    perc = np.nanpercentile(train_abs, 75) if train_abs.size else 0.0
    threshold = np.maximum(2 * cost, perc)
    opportunity = (np.abs(r_future) > threshold).astype(np.float32)
    y_trade = opportunity
    y_dir = (r_future > 0).astype(np.float32)

    mask_dir = y_trade.copy()
    y_bounce = np.zeros_like(y_dir)
    mask_bounce = np.zeros_like(y_dir)

    if USE_LEVELS_KMEANS and "near_level_flag" in df.columns:
        near_flag = df["near_level_flag"].to_numpy(dtype=np.float32)
        dist_support = df["dist_support"].to_numpy(dtype=np.float64)
        dist_resistance = df["dist_resistance"].to_numpy(dtype=np.float64)
        support_side = dist_support <= dist_resistance
        touch_tol = np.maximum(spreads, prices * LEVEL_TOL_PCT)
        bounce_thresh = LEVEL_BOUNCE_MULT * touch_tol
        break_thresh = LEVEL_BREAK_MULT * touch_tol
        max_idx = len(prices) - horizon
        for i in range(max_idx):
            if near_flag[i] <= 0:
                continue
            price = prices[i]
            if support_side[i]:
                level = price - dist_support[i]
                bounce_target = level + bounce_thresh[i]
                break_target = level - break_thresh[i]
                def _bounce_cond(p: float) -> bool:
                    return p >= bounce_target
                def _break_cond(p: float) -> bool:
                    return p <= break_target
            else:
                level = price + dist_resistance[i]
                bounce_target = level - bounce_thresh[i]
                break_target = level + break_thresh[i]
                def _bounce_cond(p: float) -> bool:
                    return p <= bounce_target
                def _break_cond(p: float) -> bool:
                    return p >= break_target

            bounce_idx = None
            break_idx = None
            window = prices[i + 1 : i + horizon + 1]
            for j, p in enumerate(window, start=1):
                if bounce_idx is None and _bounce_cond(p):
                    bounce_idx = j
                if break_idx is None and _break_cond(p):
                    break_idx = j
                if bounce_idx is not None and break_idx is not None:
                    break
            if bounce_idx is None and break_idx is None:
                continue
            mask_bounce[i] = 1.0
            if break_idx is None or (bounce_idx is not None and bounce_idx < break_idx):
                y_bounce[i] = 1.0
            else:
                y_bounce[i] = 0.0

    if not np.isfinite(y_bounce).all() or not np.isfinite(mask_bounce).all():
        _abort_nonfinite(
            "bounce_labels",
            {
                "y_bounce_finite": bool(np.isfinite(y_bounce).all()),
                "mask_bounce_finite": bool(np.isfinite(mask_bounce).all()),
            },
        )

    n = len(prices)

    y_ksl = np.zeros_like(y_trade, dtype=np.float32)
    y_ktp = np.zeros_like(y_trade, dtype=np.float32)
    mask_ksl = np.zeros_like(y_trade, dtype=np.float32)
    mask_ktp = np.zeros_like(y_trade, dtype=np.float32)
    if "vol_30m" in df.columns:
        vol_proxy = pd.to_numeric(df["vol_30m"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        vol_proxy = np.zeros_like(prices, dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64) if "high" in df.columns else prices
    low = df["low"].to_numpy(dtype=np.float64) if "low" in df.columns else prices
    horizon_sltp = max(1, int(SLTP_LABEL_HORIZON_BARS))
    eps = 1e-8
    for t in range(n):
        if y_trade[t] < 0.5:
            continue
        if t + horizon_sltp >= n:
            continue
        atr_t = float(vol_proxy[t] * prices[t]) if prices[t] > 0 else 0.0
        if atr_t <= 0:
            continue
        entry = prices[t]
        window_high = float(np.max(high[t + 1 : t + horizon_sltp + 1]))
        window_low = float(np.min(low[t + 1 : t + horizon_sltp + 1]))
        if y_dir[t] >= 0.5:
            mae = max(0.0, entry - window_low)
            mfe = max(0.0, window_high - entry)
        else:
            mae = max(0.0, window_high - entry)
            mfe = max(0.0, entry - window_low)
        k_sl_label = mae / (atr_t + eps)
        k_tp_label = mfe / (atr_t + eps)
        if CLAMP_LABELS_TOO:
            k_sl_label = _clip_if(k_sl_label, K_SL_MIN, K_SL_MAX, True)
            k_tp_label = _clip_if(k_tp_label, K_TP_MIN, K_TP_MAX, True)
        y_ksl[t] = float(k_sl_label)
        y_ktp[t] = float(k_tp_label)
        mask_ksl[t] = 1.0
        mask_ktp[t] = 1.0

    y_ttp_quick = np.zeros_like(y_trade, dtype=np.float32)
    mask_ttp = np.zeros_like(y_trade, dtype=np.float32)
    horizon_ttp = int(TTP_HORIZON_BARS) if TTP_HORIZON_BARS is not None else max(1, int(TTP_MAX_MINUTES))
    for t in range(n):
        if t + horizon_ttp >= n:
            continue
        if TTP_USE_ENTRY_ONLY_MASK and TTP_TRAIN_ON_Y_TRADE_ONLY and y_trade[t] < TTP_MIN_Y_TRADE:
            continue
        atr_t = float(vol_proxy[t] * prices[t]) if prices[t] > 0 else 0.0
        if atr_t <= 0:
            continue
        entry = prices[t]
        if TTP_DIR_LABEL_SOURCE == "existing_labels":
            side = 1 if y_dir[t] >= 0.5 else -1
        else:
            side = 1 if (prices[t + 1] - entry) >= 0 else -1
        target_dist = float(TTP_ATR_MULT) * atr_t
        if side > 0:
            hit = float(np.max(high[t + 1 : t + horizon_ttp + 1])) >= entry + target_dist
        else:
            hit = float(np.min(low[t + 1 : t + horizon_ttp + 1])) <= entry - target_dist
        y_ttp_quick[t] = 1.0 if hit else 0.0
        if TTP_LABEL_SMOOTH > 0:
            s = float(TTP_LABEL_SMOOTH)
            y_ttp_quick[t] = y_ttp_quick[t] * (1.0 - 2.0 * s) + s
        mask_ttp[t] = 1.0

    y_edge = np.zeros_like(y_trade, dtype=np.float32)
    mask_edge = np.zeros_like(y_trade, dtype=np.float32)
    y_qdist = np.zeros_like(y_trade, dtype=np.float32)
    mask_qdist = np.zeros_like(y_trade, dtype=np.float32)
    horizon_edge = max(1, int(EDGE_HORIZON_BARS))
    horizon_qdist = max(1, int(QDIST_HORIZON_BARS))
    for t in range(n):
        if t + max(horizon_edge, horizon_qdist) >= n:
            continue
        if EDGE_TRAIN_ON_Y_TRADE_ONLY and y_trade[t] < 0.5:
            continue
        atr_t = float(vol_proxy[t] * prices[t]) if prices[t] > 0 else 0.0
        if atr_t <= 0:
            continue
        entry = prices[t]
        if EDGE_DIR_LABEL_SOURCE == "existing_labels":
            side = 1 if y_dir[t] >= 0.5 else -1
        else:
            side = 1 if (prices[t + 1] - entry) >= 0 else -1
        future_edge = prices[t + horizon_edge]
        ret_edge = (future_edge - entry) if side > 0 else (entry - future_edge)
        edge_atr = ret_edge / (atr_t + EDGE_ATR_EPS)
        edge_atr = max(-EDGE_CLIP_RET_ATR, min(EDGE_CLIP_RET_ATR, float(edge_atr)))
        y_edge[t] = float(edge_atr)
        mask_edge[t] = 1.0

        if QDIST_TRAIN_ON_Y_TRADE_ONLY and y_trade[t] < 0.5:
            continue
        future_q = prices[t + horizon_qdist]
        ret_q = (future_q - entry) if side > 0 else (entry - future_q)
        q_atr = ret_q / (atr_t + EDGE_ATR_EPS)
        q_atr = max(-QDIST_CLIP_RET_ATR, min(QDIST_CLIP_RET_ATR, float(q_atr)))
        y_qdist[t] = float(q_atr)
        mask_qdist[t] = 1.0

    y_transition = np.zeros_like(y_trade, dtype=np.float32)
    mask_transition = np.zeros_like(y_trade, dtype=np.float32)
    horizon_trans = max(1, int(TRANSITION_LABEL_HORIZON_BARS))
    max_win = max(SLOPE_WINDOWS) if SLOPE_WINDOWS else 2
    for t in range(n):
        if t + horizon_trans >= n:
            continue
        if (t + 1) < max_win:
            continue
        atr_t = float(vol_proxy[t] * prices[t]) if prices[t] > 0 else 0.0
        if atr_t <= 0:
            continue
        price_hist = prices[: t + 1]
        slopes = _compute_slopes_multi_np(price_hist, SLOPE_WINDOWS)
        if SLOPE_USE_ATR_NORM:
            slopes = slopes / (atr_t + SLOPE_ATR_EPS)
        slopes = np.clip(slopes, -DISAGREE_CLIP, DISAGREE_CLIP)
        slope_short = float(slopes[0])
        slope_long = float(slopes[-1])
        if abs(slope_short) < TRANSITION_MIN_STRENGTH or abs(slope_long) < TRANSITION_MIN_STRENGTH:
            continue
        if np.sign(slope_short) == np.sign(slope_long):
            continue
        entry = prices[t]
        short_dir = 1 if slope_short > 0 else -1
        move = TRANSITION_LABEL_ATR_MOVE * atr_t
        if TRANSITION_LABEL_MODE == "direction_takeover":
            if short_dir > 0:
                hit = float(np.max(high[t + 1 : t + horizon_trans + 1])) >= entry + move
            else:
                hit = float(np.min(low[t + 1 : t + horizon_trans + 1])) <= entry - move
        else:
            future_up = float(np.max(high[t + 1 : t + horizon_trans + 1])) - entry
            future_dn = entry - float(np.min(low[t + 1 : t + horizon_trans + 1]))
            future_dir = 1 if future_up >= future_dn else -1
            hit = future_dir == short_dir
        y_transition[t] = 1.0 if hit else 0.0
        if TRANSITION_LABEL_SMOOTH > 0:
            s = float(TRANSITION_LABEL_SMOOTH)
            y_transition[t] = y_transition[t] * (1.0 - 2.0 * s) + s
        mask_transition[t] = 1.0

    horizons = tuple(int(h) for h in DIST_EXIT_HORIZONS)
    if not horizons:
        horizons = (1,)
    max_h = max(horizons)
    y_dist_targets = np.zeros((n, len(horizons)), dtype=np.float32)
    mask_dist = np.zeros(n, dtype=np.float32)
    for t in range(n):
        if t + max_h >= n:
            continue
        atr_t = float(vol_proxy[t] * prices[t]) if prices[t] > 0 else 0.0
        if atr_t <= 0:
            continue
        entry = prices[t]
        for j, h in enumerate(horizons):
            future_close = float(prices[t + h])
            r_atr = (future_close - entry) / (atr_t + 1e-8)
            r_atr = max(DIST_EXIT_CLAMP_MIN, min(DIST_EXIT_CLAMP_MAX, float(r_atr)))
            y_dist_targets[t, j] = r_atr
        mask_dist[t] = 1.0

    y_trend = np.zeros_like(y_trade, dtype=np.float32)
    y_trend_up = np.zeros_like(y_trade, dtype=np.float32)
    y_trend_up_soft = np.zeros_like(y_trade, dtype=np.float32)
    mask_trend = np.zeros_like(y_trade, dtype=np.float32)
    horizon_trend = max(1, int(TREND_HORIZON_BARS))
    trend_strength = np.zeros_like(y_trade, dtype=np.float32)
    trend_adv = np.zeros_like(y_trade, dtype=np.float32)
    for t in range(n):
        if t + horizon_trend >= n:
            continue
        atr_t = float(vol_proxy[t] * prices[t]) if prices[t] > 0 else 0.0
        if atr_t <= 0:
            continue
        entry = prices[t]
        future_up = (float(np.max(prices[t + 1 : t + horizon_trend + 1])) - entry) / (atr_t + TREND_EPS)
        future_dn = (entry - float(np.min(prices[t + 1 : t + horizon_trend + 1]))) / (atr_t + TREND_EPS)
        adv = float(future_up - future_dn)
        strength = max(future_up - TREND_LABEL_LAMBDA_DD * future_dn, future_dn - TREND_LABEL_LAMBDA_DD * future_up)
        trend_strength[t] = float(strength)
        trend_adv[t] = adv
        y_trend_up[t] = 1.0 if future_up >= future_dn else 0.0
        adv_scaled = adv / max(TREND_DIR_ADV_TEMP, 1e-6)
        adv_scaled = max(-60.0, min(60.0, adv_scaled))
        y_trend_up_soft[t] = 1.0 / (1.0 + math.exp(-adv_scaled))
        mask_trend[t] = 1.0

    if TREND_LABEL_BALANCE_MODE == "quantile":
        valid = trend_strength[mask_trend > 0]
        if valid.size:
            thr = float(np.nanquantile(valid, 1.0 - TREND_LABEL_POS_RATE))
        else:
            thr = 0.0
    else:
        thr = 0.0
    y_trend = (trend_strength > thr).astype(np.float32)
    strength_train = trend_strength[:train_end][mask_trend[:train_end] > 0]
    if strength_train.size:
        scale = float(np.nanpercentile(strength_train, 90))
    else:
        scale = 0.0
    scale = max(scale, 1e-6)
    trend_strength_norm = np.clip(trend_strength / scale, 0.0, 3.0) / 3.0

    y_break = np.zeros_like(y_trade, dtype=np.float32)
    mask_break = np.zeros_like(y_trade, dtype=np.float32)
    channel_upper = df["channel_upper"].to_numpy(dtype=np.float64) if "channel_upper" in df.columns else None
    channel_lower = df["channel_lower"].to_numpy(dtype=np.float64) if "channel_lower" in df.columns else None
    if channel_upper is not None and channel_lower is not None:
        atr_px_series = vol_proxy * prices
        high = df["high"].to_numpy(dtype=np.float64) if "high" in df.columns else prices
        low = df["low"].to_numpy(dtype=np.float64) if "low" in df.columns else prices
        horizon_break = max(1, int(BREAK_HORIZON_BARS))
        for t in range(n):
            if t + horizon_break >= n:
                continue
            upper = channel_upper[t]
            lower = channel_lower[t]
            if not np.isfinite(upper) or not np.isfinite(lower):
                continue
            if EXCEED_USE_CLOSE:
                outside_up = prices[t] > upper
                outside_dn = prices[t] < lower
                exceed_up = prices[t] - upper
                exceed_dn = lower - prices[t]
            else:
                outside_up = high[t] > upper
                outside_dn = low[t] < lower
                exceed_up = high[t] - upper
                exceed_dn = lower - low[t]
            if not (outside_up or outside_dn):
                continue
            if outside_up and outside_dn:
                if exceed_up > exceed_dn:
                    outside_dn = False
                elif exceed_dn > exceed_up:
                    outside_up = False
                else:
                    continue
            atr_t = float(atr_px_series[t]) if atr_px_series[t] > 0 else 0.0
            if atr_t <= 0:
                continue
            entry = prices[t]
            mask_break[t] = 1.0
            success = False
            if outside_up:
                target = entry + BREAK_GO_ATR * atr_t
                fail = entry - BREAK_FAIL_RETRACE_ATR * atr_t
                for j in range(1, horizon_break + 1):
                    if high[t + j] >= target:
                        success = True
                        break
                    if BREAK_FAIL_RETRACE_ATR > 0 and low[t + j] <= fail:
                        success = False
                        break
            else:
                target = entry - BREAK_GO_ATR * atr_t
                fail = entry + BREAK_FAIL_RETRACE_ATR * atr_t
                for j in range(1, horizon_break + 1):
                    if low[t + j] <= target:
                        success = True
                        break
                    if BREAK_FAIL_RETRACE_ATR > 0 and high[t + j] >= fail:
                        success = False
                        break
            y_break[t] = 1.0 if success else 0.0

    return {
        "r_future": r_future.astype(np.float32),
        "y_trade": y_trade.astype(np.float32),
        "y_dir": y_dir.astype(np.float32),
        "y_bounce": y_bounce.astype(np.float32),
        "mask_dir": mask_dir.astype(np.float32),
        "mask_bounce": mask_bounce.astype(np.float32),
        "y_ksl": y_ksl.astype(np.float32),
        "mask_ksl": mask_ksl.astype(np.float32),
        "y_ktp": y_ktp.astype(np.float32),
        "mask_ktp": mask_ktp.astype(np.float32),
        "y_ttp_quick": y_ttp_quick.astype(np.float32),
        "mask_ttp": mask_ttp.astype(np.float32),
        "y_edge": y_edge.astype(np.float32),
        "mask_edge": mask_edge.astype(np.float32),
        "y_qdist": y_qdist.astype(np.float32),
        "mask_qdist": mask_qdist.astype(np.float32),
        "y_dist_targets": y_dist_targets.astype(np.float32),
        "mask_dist": mask_dist.astype(np.float32),
        "y_trend": y_trend.astype(np.float32),
        "y_trend_up": y_trend_up.astype(np.float32),
        "y_trend_up_soft": y_trend_up_soft.astype(np.float32),
        "mask_trend": mask_trend.astype(np.float32),
        "y_transition": y_transition.astype(np.float32),
        "mask_transition": mask_transition.astype(np.float32),
        "trend_strength": trend_strength.astype(np.float32),
        "trend_strength_norm": trend_strength_norm.astype(np.float32),
        "trend_adv": trend_adv.astype(np.float32),
        "y_break": y_break.astype(np.float32),
        "mask_break": mask_break.astype(np.float32),
        "excess_up_atr": df.get("excess_up_atr", pd.Series(np.zeros_like(y_trade))).to_numpy(dtype=np.float32),
        "excess_dn_atr": df.get("excess_dn_atr", pd.Series(np.zeros_like(y_trade))).to_numpy(dtype=np.float32),
        "vol_30m": pd.to_numeric(df.get("vol_30m", 0.0), errors="coerce").to_numpy(dtype=np.float32),
        "channel_slope": pd.to_numeric(df.get("channel_slope", 0.0), errors="coerce").to_numpy(dtype=np.float32),
        "xau_spread": pd.to_numeric(df.get("xau_spread", 0.0), errors="coerce").to_numpy(dtype=np.float32),
    }


def _append_metrics(path: Path, row: dict[str, float | int | str]) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            fh.write(",".join(row.keys()) + "\n")
            fh.write(",".join(str(row[k]) for k in row.keys()) + "\n")
    else:
        with path.open("a", newline="", encoding="utf-8") as fh:
            fh.write(",".join(str(row[k]) for k in row.keys()) + "\n")


def _feature_stats(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if RET_WINDOWS:
        ret_key = f"ret_{min(RET_WINDOWS)}m"
    else:
        ret_key = "ret_1m"
    corr_ref = max(CORR_WINDOWS) if CORR_WINDOWS else None
    keys = [ret_key, "vol_30m", "channel_slope", "dist_nearest_norm"]
    if corr_ref is not None:
        keys += [f"corr_xau_xag_{corr_ref}", f"corr_xau_dxy_{corr_ref}"]
    stats = {}
    for key in keys:
        if key in df.columns:
            vals = pd.to_numeric(df[key], errors="coerce")
            stats[key] = (float(vals.min()), float(vals.max()))
    return stats


def dump_run_config_json(output_dir: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    cfg: dict[str, object] = {}
    for k, v in globals().items():
        if k.startswith("_"):
            continue
        if callable(v):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            cfg[k] = v
        elif isinstance(v, (list, tuple)) and all(
            isinstance(x, (int, float, str, bool)) or x is None for x in v
        ):
            cfg[k] = list(v)
        elif isinstance(v, dict):
            try:
                json.dumps(v)
                cfg[k] = v
            except Exception:
                pass
    cfg["run_timestamp"] = ts
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"run_config_{ts}.json"
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _compute_metrics(
    logits: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    bar_ts: np.ndarray,
) -> dict[str, float]:
    p_trade = 1.0 / (1.0 + np.exp(-logits["trade"]))
    p_up = 1.0 / (1.0 + np.exp(-logits["dir"]))
    p_bounce = 1.0 / (1.0 + np.exp(-logits["bounce"]))
    p_ttp = 1.0 / (1.0 + np.exp(-logits["ttp_quick"]))
    edge_hat = logits["edge"]
    qdist_hat = logits["qdist"]
    p_trend_gate = logits["trend"]
    p_trend_up = logits["trend_up"]
    p_transition = 1.0 / (1.0 + np.exp(-logits["transition"])) if "transition" in logits else None
    y_trade = labels["y_trade"]
    y_dir = labels["y_dir"]
    y_bounce = labels["y_bounce"]
    y_ttp = labels["y_ttp_quick"]
    y_edge = labels["y_edge"]
    y_qdist = labels["y_qdist"]
    y_trend = labels["y_trend"]
    y_trend_up = labels["y_trend_up"]
    y_trend_up_soft = labels.get("y_trend_up_soft", y_trend_up)
    y_transition = labels.get("y_transition")
    mask_dir = labels["mask_dir"] > 0
    mask_bounce = labels["mask_bounce"] > 0
    mask_ttp = labels["mask_ttp"] > 0
    mask_edge = labels["mask_edge"] > 0
    mask_qdist = labels["mask_qdist"] > 0
    mask_trend = labels["mask_trend"] > 0
    mask_transition = labels.get("mask_transition")
    trend_strength_norm = labels.get("trend_strength_norm")
    trend_adv = labels.get("trend_adv")
    y_dist_targets = labels["y_dist_targets"]
    mask_dist = labels["mask_dist"] > 0
    y_ksl = labels["y_ksl"]
    y_ktp = labels["y_ktp"]
    mask_ksl = labels["mask_ksl"] > 0
    mask_ktp = labels["mask_ktp"] > 0
    y_break = labels["y_break"]
    mask_break = labels["mask_break"] > 0
    excess_up = labels["excess_up_atr"]
    excess_dn = labels["excess_dn_atr"]

    trade_acc = float(np.mean((p_trade >= 0.5) == (y_trade >= 0.5)))
    dir_acc = float(np.mean((p_up[mask_dir] >= 0.5) == (y_dir[mask_dir] >= 0.5))) if mask_dir.any() else 0.0
    bounce_acc = (
        float(np.mean((p_bounce[mask_bounce] >= 0.5) == (y_bounce[mask_bounce] >= 0.5)))
        if mask_bounce.any()
        else 0.0
    )
    bounce_prec = 0.0
    bounce_rec = 0.0
    if mask_bounce.any():
        pred = (p_bounce[mask_bounce] >= 0.5)
        true = (y_bounce[mask_bounce] >= 0.5)
        tp = np.sum(pred & true)
        fp = np.sum(pred & ~true)
        fn = np.sum(~pred & true)
        bounce_prec = float(tp / max(tp + fp, 1))
        bounce_rec = float(tp / max(tp + fn, 1))

    ttp_acc = 0.0
    ttp_rate_pos = 0.0
    ttp_pred_mean = 0.0
    ttp_loss = 0.0
    if TTP_USE_ENTRY_ONLY_MASK and TTP_TRAIN_ON_Y_TRADE_ONLY:
        mask_ttp_entry = mask_ttp & (y_trade > TTP_MIN_Y_TRADE)
    else:
        mask_ttp_entry = mask_ttp
    if mask_ttp_entry.any():
        pred = (p_ttp[mask_ttp_entry] >= 0.5)
        true = (y_ttp[mask_ttp_entry] >= 0.5)
        ttp_acc = float(np.mean(pred == true))
        ttp_rate_pos = float(np.mean(y_ttp[mask_ttp_entry]))
        ttp_pred_mean = float(np.mean(p_ttp[mask_ttp_entry]))
        loss_raw = F.binary_cross_entropy_with_logits(
            torch.tensor(logits["ttp_quick"][mask_ttp_entry]),
            torch.tensor(y_ttp[mask_ttp_entry]),
            reduction="none",
        )
        ttp_loss = float(loss_raw.mean().item())

    edge_mae = 0.0
    edge_neg_rate = 0.0
    edge_neg_mean = 0.0
    if mask_edge.any():
        edge_err = edge_hat[mask_edge] - y_edge[mask_edge]
        edge_mae = float(np.mean(np.abs(edge_err)))
        neg_mask = y_edge[mask_edge] < 0
        edge_neg_rate = float(np.mean(neg_mask))
        if np.any(neg_mask):
            edge_neg_mean = float(np.mean(y_edge[mask_edge][neg_mask]))

    qdist_pinball = 0.0
    if mask_qdist.any():
        q_vals = np.array(QDIST_QUANTILES, dtype=np.float32).reshape(1, -1)
        yq = y_qdist[mask_qdist].reshape(-1, 1)
        pred = qdist_hat[mask_qdist]
        e = yq - pred
        pinball = np.maximum(q_vals * e, (q_vals - 1.0) * e)
        qdist_pinball = float(np.mean(pinball))

    transition_acc = 0.0
    transition_rate_pos = 0.0
    transition_pred_mean = 0.0
    transition_loss = 0.0
    if p_transition is not None and y_transition is not None and mask_transition is not None:
        mask_transition_bool = np.array(mask_transition) > 0
        if mask_transition_bool.any():
            pred = (p_transition[mask_transition_bool] >= 0.5)
            true = (y_transition[mask_transition_bool] >= 0.5)
            transition_acc = float(np.mean(pred == true))
            transition_rate_pos = float(np.mean(y_transition[mask_transition_bool]))
            transition_pred_mean = float(np.mean(p_transition[mask_transition_bool]))
            loss_raw = F.binary_cross_entropy_with_logits(
                torch.tensor(logits["transition"][mask_transition_bool]),
                torch.tensor(y_transition[mask_transition_bool]),
                reduction="none",
            )
            transition_loss = float(loss_raw.mean().item())

    trend_acc = 0.0
    trend_dir_acc = 0.0
    trend_dir_corr_adv = 0.0
    trend_dir_brier_w = 0.0
    trend_gate_pos_mean = 0.0
    trend_gate_neg_mean = 0.0
    if mask_trend.any():
        pred = (p_trend_gate[mask_trend] >= 0.5)
        true = (y_trend[mask_trend] >= 0.5)
        trend_acc = float(np.mean(pred == true))
        mask_dir = mask_trend & (y_trend >= 0.5)
        if mask_dir.any():
            pred_dir = (p_trend_up[mask_dir] >= 0.5)
            true_dir = (y_trend_up[mask_dir] >= 0.5)
            trend_dir_acc = float(np.mean(pred_dir == true_dir))
        if trend_strength_norm is not None and trend_adv is not None:
            strength_norm = np.array(trend_strength_norm, dtype=np.float32)
            adv = np.array(trend_adv, dtype=np.float32)
            w_dir = mask_trend.astype(np.float32) * strength_norm
            w_dir = w_dir * (strength_norm > TREND_DIR_MIN_STRENGTH)
            w_dir = w_dir * (1.0 / (1.0 + np.exp(-np.abs(adv) / max(TREND_DIR_ADV_CONF_TEMP, 1e-6))))
            if np.sum(w_dir) > 0:
                trend_dir_brier_w = float(
                    np.sum(w_dir * (p_trend_up - y_trend_up_soft) ** 2) / (np.sum(w_dir) + 1e-8)
                )
            mask_adv = mask_trend & (np.abs(adv) > 1e-6)
            if np.any(mask_adv):
                trend_dir_corr_adv = float(
                    np.corrcoef((p_trend_up[mask_adv] - 0.5), adv[mask_adv])[0, 1]
                )
        pos_mask = mask_trend & (y_trend >= 0.5)
        neg_mask = mask_trend & (y_trend < 0.5)
        if pos_mask.any():
            trend_gate_pos_mean = float(np.mean(p_trend_gate[pos_mask]))
        if neg_mask.any():
            trend_gate_neg_mean = float(np.mean(p_trend_gate[neg_mask]))

    coverage = float(np.mean(p_trade >= TRADE_THRESHOLD))
    conf = np.maximum(p_up, 1.0 - p_up)
    selective = {}
    for thr in (0.55, 0.60, 0.70):
        mask = (p_trade >= TRADE_THRESHOLD) & (conf >= thr)
        if mask.any():
            selective[f"selective_dir_acc_{thr}"] = float(np.mean((p_up[mask] >= 0.5) == (y_dir[mask] >= 0.5)))
        else:
            selective[f"selective_dir_acc_{thr}"] = 0.0

    trades = int(np.sum(p_trade >= TRADE_THRESHOLD))
    if len(bar_ts) > 1:
        ts = pd.to_datetime(bar_ts, utc=True, errors="coerce")
        days = (ts[-1] - ts[0]).total_seconds() / 86400.0 if not pd.isna(ts[-1]) else 0.0
    else:
        days = 0.0
    trades_per_day = float(trades / max(days, 1.0))

    break_acc = 0.0
    break_prec = 0.0
    break_rec = 0.0
    break_mask_rate = float(np.mean(mask_break)) if mask_break.size else 0.0
    if mask_break.any():
        pred_break = logits["break"][mask_break] >= 0.5
        true_break = y_break[mask_break] >= 0.5
        break_acc = float(np.mean(pred_break == true_break))
        tp = np.sum(pred_break & true_break)
        fp = np.sum(pred_break & ~true_break)
        fn = np.sum(~pred_break & true_break)
        break_prec = float(tp / max(tp + fp, 1))
        break_rec = float(tp / max(tp + fn, 1))

    ksl_mae = 0.0
    ksl_rmse = 0.0
    ksl_corr = 0.0
    ksl_coverage = float(np.mean(mask_ksl)) if mask_ksl.size else 0.0
    ksl_y_mean = 0.0
    ksl_y_p90 = 0.0
    if mask_ksl.any():
        ksl_pred = logits["ksl"][mask_ksl]
        ksl_true = y_ksl[mask_ksl]
        err = ksl_pred - ksl_true
        ksl_mae = float(np.mean(np.abs(err)))
        ksl_rmse = float(np.sqrt(np.mean(err**2)))
        if ksl_pred.size > 2 and np.std(ksl_true) > 0 and np.std(ksl_pred) > 0:
            ksl_corr = float(np.corrcoef(ksl_pred, ksl_true)[0, 1])
        ksl_y_mean = float(np.mean(ksl_true))
        ksl_y_p90 = float(np.nanpercentile(ksl_true, 90))

    ktp_mae = 0.0
    ktp_rmse = 0.0
    ktp_corr = 0.0
    ktp_coverage = float(np.mean(mask_ktp)) if mask_ktp.size else 0.0
    ktp_y_mean = 0.0
    ktp_y_p90 = 0.0
    if mask_ktp.any():
        ktp_pred = logits["ktp"][mask_ktp]
        ktp_true = y_ktp[mask_ktp]
        err = ktp_pred - ktp_true
        ktp_mae = float(np.mean(np.abs(err)))
        ktp_rmse = float(np.sqrt(np.mean(err**2)))
        if ktp_pred.size > 2 and np.std(ktp_true) > 0 and np.std(ktp_pred) > 0:
            ktp_corr = float(np.corrcoef(ktp_pred, ktp_true)[0, 1])
        ktp_y_mean = float(np.mean(ktp_true))
        ktp_y_p90 = float(np.nanpercentile(ktp_true, 90))

    dist_mae_q50_h5 = 0.0
    dist_pinball_mean = 0.0
    if mask_dist.any():
        dist_pred = logits["dist_exit"][mask_dist]
        dist_true = y_dist_targets[mask_dist]
        h_idx = _horizon_index(DIST_EXIT_EV_HORIZON)
        q50_idx = _quantile_index(50)
        pred_q50 = dist_pred[:, h_idx, q50_idx]
        true_h = dist_true[:, h_idx]
        dist_mae_q50_h5 = float(np.mean(np.abs(pred_q50 - true_h)))

        q_vals = np.array(DIST_EXIT_QUANTILES, dtype=np.float32) / 100.0
        q_vals = q_vals.reshape(1, 1, -1)
        e = dist_true[:, :, None] - dist_pred
        pinball = np.maximum(q_vals * e, (q_vals - 1.0) * e)
        dist_pinball_mean = float(np.mean(pinball))

    breakout_bins = {}
    if mask_break.any():
        excess = np.maximum(excess_up, excess_dn)
        edges = [0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0]
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask_bin = mask_break & (excess >= lo) & (excess < hi)
            key = f"{lo}_{hi}".replace(".", "p")
            if mask_bin.any():
                breakout_bins[f"break_success_rate_bin_{key}"] = float(np.mean(y_break[mask_bin]))
                breakout_bins[f"break_p_break_mean_bin_{key}"] = float(np.mean(logits["break"][mask_bin]))
            else:
                breakout_bins[f"break_success_rate_bin_{key}"] = 0.0
                breakout_bins[f"break_p_break_mean_bin_{key}"] = 0.0

    return {
        "trade_acc": trade_acc,
        "dir_acc_on_trades": dir_acc,
        "bounce_acc_on_near": bounce_acc,
        "bounce_precision_on_near": bounce_prec,
        "bounce_recall_on_near": bounce_rec,
        "coverage": coverage,
        "trades_per_day": trades_per_day,
        **selective,
        "p_trade_mean": float(np.mean(p_trade)),
        "p_up_mean": float(np.mean(p_up)),
        "p_bounce_mean": float(np.mean(p_bounce)),
        "p_ttp_mean": float(np.mean(p_ttp)),
        "p_trend_gate_mean": float(np.mean(p_trend_gate)),
        "p_trend_up_mean": float(np.mean(p_trend_up)),
        "trend_rate": float(np.mean(y_trend)),
        "opportunity_rate": float(np.mean(y_trade)),
        "near_level_rate": float(np.mean(labels["mask_bounce"])),
        "bounce_mask_rate": float(np.mean(labels["mask_bounce"])),
        "bounce_rate": float(np.mean(y_bounce[mask_bounce])) if mask_bounce.any() else 0.0,
        "break_acc_on_mask": break_acc,
        "break_precision_on_mask": break_prec,
        "break_recall_on_mask": break_rec,
        "break_mask_rate": break_mask_rate,
        "ksl_mae": ksl_mae,
        "ksl_rmse": ksl_rmse,
        "ksl_corr": ksl_corr,
        "ksl_coverage": ksl_coverage,
        "ktp_mae": ktp_mae,
        "ktp_rmse": ktp_rmse,
        "ktp_corr": ktp_corr,
        "ktp_coverage": ktp_coverage,
        "y_ksl_mean": ksl_y_mean,
        "y_ksl_p90": ksl_y_p90,
        "y_ktp_mean": ktp_y_mean,
        "y_ktp_p90": ktp_y_p90,
        "ttp_loss": ttp_loss,
        "ttp_acc": ttp_acc,
        "ttp_rate_pos": ttp_rate_pos,
        "ttp_pred_mean": ttp_pred_mean,
        "edge_mae": edge_mae,
        "edge_neg_rate": edge_neg_rate,
        "edge_neg_mean": edge_neg_mean,
        "qdist_pinball": qdist_pinball,
        "transition_loss": transition_loss,
        "transition_acc": transition_acc,
        "transition_rate_pos": transition_rate_pos,
        "transition_pred_mean": transition_pred_mean,
        "trend_acc": trend_acc,
        "trend_dir_acc": trend_dir_acc,
        "trend_dir_corr_adv": trend_dir_corr_adv,
        "trend_dir_brier_w": trend_dir_brier_w,
        "trend_gate_pos_mean": trend_gate_pos_mean,
        "trend_gate_neg_mean": trend_gate_neg_mean,
        "dist_mae_q50_h5": dist_mae_q50_h5,
        "dist_pinball_mean": dist_pinball_mean,
        **breakout_bins,
    }


def _masked_stats_probs(p: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    p = p[mask]
    if p.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p01_frac": 0.0,
            "p99_frac": 0.0,
            "p1": 0.0,
            "p5": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    percentiles = np.percentile(p, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "count": float(p.size),
        "mean": float(np.mean(p)),
        "min": float(np.min(p)),
        "max": float(np.max(p)),
        "p01_frac": float(np.mean(p < 1e-4)),
        "p99_frac": float(np.mean(p > 1.0 - 1e-4)),
        "p1": float(percentiles[0]),
        "p5": float(percentiles[1]),
        "p10": float(percentiles[2]),
        "p25": float(percentiles[3]),
        "p50": float(percentiles[4]),
        "p75": float(percentiles[5]),
        "p90": float(percentiles[6]),
        "p95": float(percentiles[7]),
        "p99": float(percentiles[8]),
    }


def _masked_class_balance(y: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    y = y[mask]
    if y.size == 0:
        return {"count": 0.0, "pos_rate": 0.0}
    return {"count": float(y.size), "pos_rate": float(np.mean(y))}


def _masked_accuracy(p: np.ndarray, y: np.ndarray, mask: np.ndarray, thr: float = 0.5) -> float:
    p = p[mask]
    y = y[mask]
    if p.size == 0:
        return 0.0
    return float(np.mean((p >= thr) == (y >= 0.5)))


def _masked_brier(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    p = p[mask]
    y = y[mask]
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def _masked_auc(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    p = p[mask]
    y = y[mask]
    if p.size == 0:
        return float("nan")
    y = (y >= 0.5).astype(np.int64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, p.size + 1)
    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


class _DiagCollector:
    def __init__(self, max_rows: int, fmt: str, out_dir: str, split: str) -> None:
        self.max_rows = int(max_rows)
        self.fmt = fmt
        self.out_dir = Path(out_dir)
        self.split = split
        self.rows: list[dict[str, object]] = []
        self.total_seen = 0
        self.rng = np.random.default_rng(42)

    def add_rows(self, rows: list[dict[str, object]]) -> None:
        if self.max_rows <= 0:
            return
        for row in rows:
            self.total_seen += 1
            if len(self.rows) < self.max_rows:
                self.rows.append(row)
            else:
                j = int(self.rng.integers(0, self.total_seen))
                if j < self.max_rows:
                    self.rows[j] = row

    def write(self, epoch: int) -> Optional[Path]:
        if not self.rows:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"diag_epoch_{epoch}_{self.split}.{self.fmt}"
        path = self.out_dir / filename
        if self.fmt == "csv":
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(self.rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.rows)
        else:
            cols = {k: np.array([r.get(k) for r in self.rows]) for k in self.rows[0].keys()}
            cols["schema"] = np.array(list(self.rows[0].keys()))
            np.savez_compressed(path, **cols)
        return path


def _trend_diag_single(
    close_short_seq: np.ndarray,
    close_long_seq: np.ndarray,
    trend_short_seq: Optional[np.ndarray] = None,
) -> dict[str, float]:
    return _trend_diag_from_seq(close_short_seq, close_long_seq, trend_short_seq=trend_short_seq)


def _run_epoch(
    model: TrendMRModel,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    split_name: str,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    diag_collector: Optional[_DiagCollector] = None,
) -> tuple[float, dict[str, np.ndarray]]:
    train = optimizer is not None
    total_loss = 0.0
    total_steps = 0
    loss_parts = {
        "trade": 0.0,
        "dir": 0.0,
        "bounce": 0.0,
        "ttp": 0.0,
        "edge": 0.0,
        "qdist": 0.0,
        "trend": 0.0,
        "trend_dir": 0.0,
        "transition": 0.0,
        "combiner": 0.0,
        "ksl": 0.0,
        "ktp": 0.0,
        "dist_exit": 0.0,
        "break": 0.0,
    }
    diag = {
        "trend_dir_count": 0,
        "trend_dir_pos_sum": 0.0,
        "trend_dir_acc_sum": 0.0,
        "trend_dir_brier_sum": 0.0,
        "trend_dir_samples": 0,
        "trend_dir_empty_batches": 0,
        "trend_dir_grad_norm": 0.0,
        "trend_dir_grad_steps": 0,
        "trend_dir_preds": [],
        "trend_dir_labels": [],
    }
    max_diag_samples = 20000
    rng = np.random.default_rng(42)
    grad_norms: list[float] = []
    logits = {
        "trade": [],
        "dir": [],
        "bounce": [],
        "ttp_quick": [],
        "edge": [],
        "qdist": [],
        "trend": [],
        "trend_up": [],
        "transition": [],
        "combiner": [],
        "ksl": [],
        "ktp": [],
        "dist_exit": [],
        "break": [],
    }
    labels = {
        "y_trade": [],
        "y_dir": [],
        "y_bounce": [],
        "y_ttp_quick": [],
        "y_edge": [],
        "mask_edge": [],
        "y_qdist": [],
        "mask_qdist": [],
        "y_trend": [],
        "y_trend_up": [],
        "y_trend_up_soft": [],
        "y_transition": [],
        "trend_strength_norm": [],
        "trend_adv": [],
        "y_combiner": [],
        "mask_combiner": [],
        "mask_dir": [],
        "mask_bounce": [],
        "mask_ttp": [],
        "mask_edge": [],
        "mask_qdist": [],
        "mask_trend": [],
        "mask_transition": [],
        "y_ksl": [],
        "mask_ksl": [],
        "y_ktp": [],
        "mask_ktp": [],
        "y_dist_targets": [],
        "mask_dist": [],
        "y_break": [],
        "mask_break": [],
        "excess_up_atr": [],
        "excess_dn_atr": [],
    }
    bar_ts = []

    bce_trade = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor([POS_W_TRADE], device=DEVICE)
    )
    bce_dir = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor([POS_W_DIR], device=DEVICE)
    )
    bce_bounce = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor([POS_W_BOUNCE], device=DEVICE)
    )
    for batch in loader:
        short = batch["short"].to(DEVICE)
        mid = batch["mid"].to(DEVICE)
        long = batch["long"].to(DEVICE)
        trend_short = batch["trend_short"].to(DEVICE)
        trend_long = batch["trend_long"].to(DEVICE)
        trend_close_short = batch["trend_close_short"].to(DEVICE)
        trend_close_long = batch["trend_close_long"].to(DEVICE)
        slow = batch["slow"].to(DEVICE)
        y_trade = batch["y_trade"].to(DEVICE)
        y_dir = batch["y_dir"].to(DEVICE)
        y_bounce = batch["y_bounce"].to(DEVICE)
        y_trend = batch["y_trend"].to(DEVICE)
        y_trend_up = batch["y_trend_up"].to(DEVICE)
        y_trend_up_soft = batch["y_trend_up_soft"].to(DEVICE)
        y_transition = batch["y_transition"].to(DEVICE)
        y_ttp_quick = batch["y_ttp_quick"].to(DEVICE)
        y_edge = batch["y_edge"].to(DEVICE)
        y_qdist = batch["y_qdist"].to(DEVICE)
        mask_dir = batch["mask_dir"].to(DEVICE)
        mask_bounce = batch["mask_bounce"].to(DEVICE)
        mask_trend = batch["mask_trend"].to(DEVICE)
        mask_transition = batch["mask_transition"].to(DEVICE)
        mask_ttp = batch["mask_ttp"].to(DEVICE)
        mask_edge = batch["mask_edge"].to(DEVICE)
        mask_qdist = batch["mask_qdist"].to(DEVICE)
        y_ksl = batch["y_ksl"].to(DEVICE)
        mask_ksl = batch["mask_ksl"].to(DEVICE)
        y_ktp = batch["y_ktp"].to(DEVICE)
        mask_ktp = batch["mask_ktp"].to(DEVICE)
        y_dist_targets = batch["y_dist_targets"].to(DEVICE)
        mask_dist = batch["mask_dist"].to(DEVICE)
        y_break = batch["y_break"].to(DEVICE)
        mask_break = batch["mask_break"].to(DEVICE)
        excess_up_atr = batch["excess_up_atr"].to(DEVICE)
        excess_dn_atr = batch["excess_dn_atr"].to(DEVICE)
        trend_strength = batch.get("trend_strength")
        trend_strength_norm = batch.get("trend_strength_norm")
        trend_adv = batch.get("trend_adv")
        vol_30m = batch.get("vol_30m")
        channel_slope = batch.get("channel_slope")
        xau_spread = batch.get("xau_spread")

        if train:
            model.train()
        else:
            model.eval()

        with torch.set_grad_enabled(train):
            (
                trade_logit,
                dir_logit,
                bounce_logit,
                ttp_logit,
                edge_pred,
                qdist_pred,
                p_trend_gate,
                ksl_pred,
                ktp_pred,
                dist_pred,
                break_pred,
                p_trend_up,
                transition_logit,
            ) = model(short, mid, long, slow, trend_short, trend_long, trend_close_short, trend_close_long)
            loss_trade = bce_trade(trade_logit, y_trade).mean()

            dir_loss_raw = bce_dir(dir_logit, y_dir)
            if mask_dir.sum() > 0:
                loss_dir = (dir_loss_raw * mask_dir).sum() / (mask_dir.sum() + 1e-8)
            else:
                loss_dir = torch.tensor(0.0, device=DEVICE)

            bounce_loss_raw = bce_bounce(bounce_logit, y_bounce)
            if mask_bounce.sum() > 0:
                loss_bounce = (bounce_loss_raw * mask_bounce).sum() / (mask_bounce.sum() + 1e-8)
            else:
                loss_bounce = torch.tensor(0.0, device=DEVICE)

            loss_ksl = torch.tensor(0.0, device=DEVICE)
            loss_ktp = torch.tensor(0.0, device=DEVICE)
            loss_dist = torch.tensor(0.0, device=DEVICE)
            loss_break = torch.tensor(0.0, device=DEVICE)
            loss_ttp = torch.tensor(0.0, device=DEVICE)
            loss_edge = torch.tensor(0.0, device=DEVICE)
            loss_qdist = torch.tensor(0.0, device=DEVICE)
            loss_trend = torch.tensor(0.0, device=DEVICE)
            loss_trend_dir = torch.tensor(0.0, device=DEVICE)
            loss_transition = torch.tensor(0.0, device=DEVICE)
            loss_combiner = torch.tensor(0.0, device=DEVICE)
            if LEARNED_SLTP:
                smooth_l1 = nn.SmoothL1Loss(reduction="none")
                loss_ksl_raw = smooth_l1(ksl_pred, y_ksl)
                loss_ktp_raw = smooth_l1(ktp_pred, y_ktp)
                loss_ksl = (loss_ksl_raw * mask_ksl).sum() / (mask_ksl.sum() + 1e-8)
                loss_ktp = (loss_ktp_raw * mask_ktp).sum() / (mask_ktp.sum() + 1e-8)
            if USE_TTP_HEAD:
                ttp_loss_raw = F.binary_cross_entropy_with_logits(
                    ttp_logit,
                    y_ttp_quick,
                    reduction="none",
                )
                if TTP_USE_ENTRY_ONLY_MASK and TTP_TRAIN_ON_Y_TRADE_ONLY:
                    entry_mask = (y_trade > TTP_MIN_Y_TRADE).float()
                    mask_ttp_entry = mask_ttp * entry_mask
                else:
                    mask_ttp_entry = mask_ttp
                if mask_ttp_entry.sum() > 0:
                    loss_ttp = (ttp_loss_raw * mask_ttp_entry).sum() / (mask_ttp_entry.sum() + 1e-8)
                else:
                    loss_ttp = ttp_logit.mean() * 0.0
            if USE_EDGE_HEAD:
                edge_mask = mask_edge
                edge_err = edge_pred - y_edge
                base = F.smooth_l1_loss(edge_pred, y_edge, reduction="none")
                if USE_RISK_AWARE_LOSS and RISK_AWARE_MODE in ("edge", "both"):
                    w = torch.where(y_edge < 0, DOWNSIDE_MULT, UPSIDE_MULT)
                    base = base * w
                if edge_mask.sum() > 0:
                    loss_edge = (base * edge_mask).sum() / (edge_mask.sum() + 1e-8)
                else:
                    loss_edge = edge_pred.mean() * 0.0
            if USE_QDIST_HEAD:
                q = torch.tensor(QDIST_QUANTILES, device=DEVICE, dtype=torch.float32)
                yq = y_qdist.unsqueeze(-1)
                e = yq - qdist_pred
                pinball = torch.maximum(q * e, (q - 1.0) * e)
                tail_w = torch.ones_like(q)
                if q.numel() >= 2:
                    tail_w[0] = QDIST_TAIL_EXTRA_WEIGHT
                    tail_w[-1] = QDIST_TAIL_EXTRA_WEIGHT
                pinball = pinball * tail_w
                if USE_RISK_AWARE_LOSS and RISK_AWARE_MODE in ("qdistrict", "both"):
                    w = torch.where(y_qdist < 0, DOWNSIDE_MULT, UPSIDE_MULT).unsqueeze(-1)
                    pinball = pinball * w
                if mask_qdist.sum() > 0:
                    loss_qdist = (pinball * mask_qdist.unsqueeze(-1)).sum() / (
                        mask_qdist.sum() * q.numel() + 1e-8
                    )
                else:
                    loss_qdist = qdist_pred.mean() * 0.0
                if QDIST_MONOTONIC_PENALTY_W > 0 and qdist_pred.shape[-1] >= 2:
                    penalty = F.relu(qdist_pred[:, 0] - qdist_pred[:, 1]).mean()
                    for k in range(1, qdist_pred.shape[-1] - 1):
                        penalty = penalty + F.relu(qdist_pred[:, k] - qdist_pred[:, k + 1]).mean()
                    loss_qdist = loss_qdist + QDIST_MONOTONIC_PENALTY_W * penalty
            if mask_trend.sum() > 0:
                trend_loss_raw = F.binary_cross_entropy(
                    p_trend_gate,
                    y_trend,
                    reduction="none",
                )
                trend_weights = torch.where(y_trend > 0.5, POS_W_TREND, 1.0)
                loss_trend = (trend_loss_raw * trend_weights * mask_trend).sum() / (mask_trend.sum() + 1e-8)
                if trend_strength_norm is not None:
                    strength_norm = trend_strength_norm.to(DEVICE)
                else:
                    strength_norm = torch.zeros_like(y_trend)
                if trend_adv is not None:
                    adv = trend_adv.to(DEVICE)
                else:
                    adv = torch.zeros_like(y_trend)
                w_dir = mask_trend * strength_norm
                w_dir = w_dir * (strength_norm > TREND_DIR_MIN_STRENGTH).float()
                w_dir = w_dir * torch.sigmoid(torch.abs(adv) / max(TREND_DIR_ADV_CONF_TEMP, 1e-6))
                p_dir = p_trend_up.clamp(1e-4, 1.0 - 1e-4)
                trend_dir_loss_raw = F.binary_cross_entropy(p_dir, y_trend_up_soft, reduction="none")
                if w_dir.sum() > 0:
                    loss_trend_dir = (trend_dir_loss_raw * w_dir).sum() / (w_dir.sum() + 1e-8)
                else:
                    loss_trend_dir = p_trend_up.mean() * 0.0
                with torch.no_grad():
                    mask_bool = (w_dir > 0)
                    if mask_bool.sum().item() == 0:
                        diag["trend_dir_empty_batches"] += 1
                    else:
                        p_sel = p_trend_up[mask_bool].detach().cpu().numpy()
                        y_sel = y_trend_up[mask_bool].detach().cpu().numpy()
                        diag["trend_dir_count"] += int(p_sel.size)
                        diag["trend_dir_pos_sum"] += float(np.sum(y_sel))
                        diag["trend_dir_acc_sum"] += float(np.sum((p_sel >= 0.5) == (y_sel >= 0.5)))
                        diag["trend_dir_brier_sum"] += float(np.sum((p_sel - y_sel) ** 2))
                        diag["trend_dir_samples"] += int(p_sel.size)
                        remain = max_diag_samples - len(diag["trend_dir_preds"])
                        if remain > 0:
                            take = min(remain, p_sel.size)
                            diag["trend_dir_preds"].extend(p_sel[:take].tolist())
                            diag["trend_dir_labels"].extend(y_sel[:take].tolist())
                        else:
                            if p_sel.size > 0 and rng.random() < max_diag_samples / max(1, diag["trend_dir_count"]):
                                replace_idx = int(rng.integers(0, max_diag_samples))
                                diag["trend_dir_preds"][replace_idx] = float(p_sel[0])
                                diag["trend_dir_labels"][replace_idx] = float(y_sel[0])
            if USE_TRANSITION_HEAD:
                pos_weight = None
                if USE_POS_W_TRANSITION:
                    pos_weight = torch.tensor([POS_W_TRANSITION], device=DEVICE)
                transition_loss_raw = F.binary_cross_entropy_with_logits(
                    transition_logit,
                    y_transition,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                if mask_transition.sum() > 0:
                    loss_transition = (transition_loss_raw * mask_transition).sum() / (mask_transition.sum() + 1e-8)
                else:
                    loss_transition = transition_logit.mean() * 0.0
            if USE_LEARNED_ENTRY_COMBINER:
                trade_feat = trade_logit if COMBINER_USE_LOGITS else torch.sigmoid(trade_logit)
                trend_feat = _logit_from_prob(p_trend_gate) if COMBINER_USE_LOGITS else p_trend_gate
                trend_dir_feat = _logit_from_prob(p_trend_up) if COMBINER_USE_LOGITS else p_trend_up
                dir_feat = dir_logit if COMBINER_USE_LOGITS else torch.sigmoid(dir_logit)
                feats = [trade_feat, trend_feat, trend_dir_feat, dir_feat]
                if COMBINER_INCLUDE_TTP:
                    ttp_feat = ttp_logit if COMBINER_USE_LOGITS else torch.sigmoid(ttp_logit)
                    ttp_feat = torch.clamp(ttp_feat, -COMBINER_TTP_CLIP, COMBINER_TTP_CLIP)
                    feats.append(ttp_feat)
                if COMBINER_INCLUDE_EDGE:
                    edge_feat = edge_pred
                    edge_feat = torch.clamp(edge_feat, -COMBINER_EDGE_CLIP, COMBINER_EDGE_CLIP)
                    feats.append(edge_feat)
                if COMBINER_INCLUDE_QDIST:
                    qdist_feat = torch.clamp(qdist_pred, -COMBINER_QDIST_CLIP, COMBINER_QDIST_CLIP)
                    feats.extend([qdist_feat[:, i] for i in range(qdist_feat.shape[1])])
                if COMBINER_INCLUDE_TAIL:
                    tail_asym = qdist_pred[:, -1] - qdist_pred[:, 0]
                    tail_asym = torch.clamp(tail_asym, -COMBINER_TAIL_CLIP, COMBINER_TAIL_CLIP)
                    feats.append(tail_asym)
                if COMBINER_INCLUDE_CONTEXT:
                    vol_feat = vol_30m.to(DEVICE) if vol_30m is not None else torch.zeros_like(trade_feat)
                    spread_feat = xau_spread.to(DEVICE) if xau_spread is not None else torch.zeros_like(trade_feat)
                    strength_feat = (
                        trend_strength_norm.to(DEVICE) if trend_strength_norm is not None else torch.zeros_like(trade_feat)
                    )
                    vol_feat = torch.clamp(vol_feat, 0.0, 10.0)
                    spread_feat = torch.clamp(spread_feat, 0.0, 100.0)
                    strength_feat = torch.clamp(strength_feat, 0.0, 1.0)
                    feats.extend([vol_feat, spread_feat, strength_feat])
                comb_in = torch.stack(feats, dim=-1)
                combiner_logit = model.entry_combiner(comb_in)
                if COMBINER_TRAIN_MASK_MODE == "trade_and_ttp_mask":
                    mask_comb = mask_ttp
                else:
                    mask_comb = (y_trade > 0.0).float()
                base = (y_trade > 0.5).float()
                quick = y_ttp_quick.float()
                if COMBINER_TARGET_MODE == "trade_and_ttp":
                    if COMBINER_USE_SOFT_TARGET:
                        y_eff = base * (COMBINER_TTP_FLOOR + (1.0 - COMBINER_TTP_FLOOR) * quick)
                    else:
                        y_eff = base * quick
                else:
                    y_eff = base
                if COMBINER_LABEL_SMOOTH > 0:
                    s = float(COMBINER_LABEL_SMOOTH)
                    y_eff = y_eff * (1.0 - 2.0 * s) + s
                if mask_comb.sum() > 0:
                    comb_loss_raw = F.binary_cross_entropy_with_logits(
                        combiner_logit[mask_comb > 0],
                        y_eff[mask_comb > 0],
                        reduction="none",
                    )
                    loss_combiner = comb_loss_raw.mean()
                else:
                    loss_combiner = combiner_logit.mean() * 0.0
            if DIST_EXIT_ENABLE:
                q = torch.tensor(DIST_EXIT_QUANTILES, device=DEVICE, dtype=torch.float32) / 100.0
                y = y_dist_targets.unsqueeze(-1)
                e = y - dist_pred
                pinball = torch.maximum(q * e, (q - 1.0) * e)
                mask = mask_dist.view(-1, 1, 1)
                loss_dist = (pinball * mask).sum() / (mask.sum() * len(DIST_EXIT_HORIZONS) * len(DIST_EXIT_QUANTILES) + 1e-8)
            if ENABLE_BREAKOUT_HEAD:
                bce_break = nn.BCELoss(reduction="none")
                loss_break_raw = bce_break(break_pred, y_break)
                weights = torch.where(y_break > 0.5, POS_W_BREAK, 1.0) * mask_break
                loss_break = (loss_break_raw * weights).sum() / (weights.sum() + 1e-8)

            if diag_collector is not None:
                with torch.no_grad():
                    mask_trend_dir = (mask_trend > 0)
                    p_trend_up_np = p_trend_up.detach().cpu().numpy()
                    p_trend_np = p_trend_gate.detach().cpu().numpy()
                    y_trend_up_np = y_trend.detach().cpu().numpy()
                    y_trend_np = y_trend.detach().cpu().numpy()
                    y_trend_up_soft_np = y_trend_up_soft.detach().cpu().numpy()
                    mask_trend_np = (mask_trend.detach().cpu().numpy() > 0).astype(np.int64)
                    mask_trend_dir_np = mask_trend_dir.detach().cpu().numpy().astype(np.int64)
                    trend_dir_loss_raw = F.binary_cross_entropy(
                        p_trend_up, y_trend_up_soft, reduction="none"
                    ).detach().cpu().numpy()

                    ttp_logit_np = ttp_logit.detach().cpu().numpy()
                    p_ttp_np = torch.sigmoid(ttp_logit).detach().cpu().numpy()
                    y_ttp_quick_np = y_ttp_quick.detach().cpu().numpy()
                    mask_ttp_np = (mask_ttp.detach().cpu().numpy() > 0).astype(np.int64)
                    entry_mask_np = (y_trade.detach().cpu().numpy() > 0.5).astype(np.int64)
                    mask_ttp_entry_np = mask_ttp_np * entry_mask_np if TTP_USE_ENTRY_ONLY_MASK else mask_ttp_np
                    ttp_loss_raw = F.binary_cross_entropy_with_logits(
                        ttp_logit, y_ttp_quick, reduction="none"
                    ).detach().cpu().numpy()

                    trend_strength_np = trend_strength.detach().cpu().numpy() if trend_strength is not None else None
                    trend_strength_norm_np = (
                        trend_strength_norm.detach().cpu().numpy() if trend_strength_norm is not None else None
                    )
                    trend_adv_np = trend_adv.detach().cpu().numpy() if trend_adv is not None else None
                    vol_np = vol_30m.detach().cpu().numpy() if vol_30m is not None else None
                    slope_np = channel_slope.detach().cpu().numpy() if channel_slope is not None else None
                    spread_np = xau_spread.detach().cpu().numpy() if xau_spread is not None else None
                    bar_ts_np = np.array(batch["bar_ts"])

                    rows: list[dict[str, object]] = []
                    for i in range(p_trend_up_np.shape[0]):
                        ts_val = bar_ts_np[i]
                        try:
                            ts_str = pd.to_datetime(ts_val, utc=True, errors="coerce")
                            ts_out = ts_str.isoformat() if ts_str is not pd.NaT else ""
                            hour = int(ts_str.hour) if ts_str is not pd.NaT else -1
                        except Exception:
                            ts_out = ""
                            hour = -1
                        trend_diag = _trend_diag_single(
                            trend_close_short[i].detach().cpu().numpy(),
                            trend_close_long[i].detach().cpu().numpy(),
                            trend_short_seq=trend_short[i].detach().cpu().numpy(),
                        )
                        row = {
                            "epoch": epoch,
                            "split": split_name,
                            "ts": ts_out,
                            "hour": hour,
                            "mask_trend": int(mask_trend_np[i]),
                            "y_trend": float(y_trend_np[i]),
                            "mask_trend_dir": int(mask_trend_dir_np[i]),
                            "y_trend_up": float(y_trend_up_np[i]),
                            "y_trend_up_soft": float(y_trend_up_soft_np[i]),
                            "p_trend": float(p_trend_np[i]),
                            "p_trend_up": float(p_trend_up_np[i]),
                            "trend_dir_loss_raw": float(trend_dir_loss_raw[i]),
                            "trend_strength": float(trend_strength_np[i]) if trend_strength_np is not None else float("nan"),
                            "trend_strength_norm": float(trend_strength_norm_np[i])
                            if trend_strength_norm_np is not None
                            else float("nan"),
                            "trend_adv": float(trend_adv_np[i]) if trend_adv_np is not None else float("nan"),
                            "mask_ttp": int(mask_ttp_np[i]),
                            "mask_ttp_entry": int(mask_ttp_entry_np[i]),
                            "y_ttp_quick": float(y_ttp_quick_np[i]),
                            "ttp_logit": float(ttp_logit_np[i]),
                            "p_ttp": float(p_ttp_np[i]),
                            "ttp_loss_raw": float(ttp_loss_raw[i]),
                            "vol_30m": float(vol_np[i]) if vol_np is not None else float("nan"),
                            "channel_slope": float(slope_np[i]) if slope_np is not None else float("nan"),
                            "spread": float(spread_np[i]) if spread_np is not None else float("nan"),
                            "push_up_atr": float(trend_diag.get("push_up_atr", float("nan"))),
                            "push_dn_atr": float(trend_diag.get("push_dn_atr", float("nan"))),
                            "pullback_dn_atr": float(trend_diag.get("pullback_dn_atr", float("nan"))),
                            "pullback_up_atr": float(trend_diag.get("pullback_up_atr", float("nan"))),
                            "ratio_up": float(trend_diag.get("ratio_up", float("nan"))),
                            "ratio_dn": float(trend_diag.get("ratio_dn", float("nan"))),
                        }
                        rows.append(row)
                    diag_collector.add_rows(rows)

            # Direction is trained as soft advantage, weighted by trend strength; TTP is entry-conditioned.
            loss = (
                LOSS_W_TRADE * loss_trade
                + LOSS_W_DIR * DIR_LOSS_WEIGHT * loss_dir
                + LOSS_W_BOUNCE * loss_bounce
            )
            if LEARNED_SLTP:
                loss = loss + LOSS_W_KSL * loss_ksl + LOSS_W_KTP * loss_ktp
            if USE_TTP_HEAD:
                loss = loss + LOSS_W_TTP * loss_ttp
            if USE_EDGE_HEAD:
                loss = loss + LOSS_W_EDGE * loss_edge
            if USE_QDIST_HEAD:
                loss = loss + LOSS_W_QDIST * loss_qdist
            loss = loss + LOSS_W_TREND * loss_trend + LOSS_W_TREND_DIR * loss_trend_dir
            if USE_TRANSITION_HEAD:
                loss = loss + LOSS_W_TRANSITION * loss_transition
            if DIST_EXIT_ENABLE:
                loss = loss + DIST_EXIT_LOSS_W * loss_dist
            if ENABLE_BREAKOUT_HEAD:
                loss = loss + LOSS_W_BREAK * loss_break
            if USE_LEARNED_ENTRY_COMBINER:
                loss = loss + LOSS_W_COMBINER * loss_combiner

            if not torch.isfinite(loss).item():
                _abort_nonfinite("loss_nonfinite", {"loss": float(loss.detach().cpu().item())})

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                grad_norms.append(float(grad_norm))
                trend_params = list(model.trend_gate.parameters())
                if trend_params:
                    total = 0.0
                    count = 0
                    for p in trend_params:
                        if p.grad is not None:
                            total += float(p.grad.detach().pow(2).sum().item())
                            count += 1
                    if count > 0:
                        diag["trend_dir_grad_norm"] += float(math.sqrt(total))
                        diag["trend_dir_grad_steps"] += 1
                if not math.isfinite(float(grad_norm)):
                    _abort_nonfinite("grad_nonfinite", {"grad_norm": float(grad_norm)})
                optimizer.step()

        step_loss = float(loss.detach().cpu().item())
        total_loss += step_loss
        total_steps += 1
        if writer is not None:
            global_step = (epoch - 1) * len(loader) + total_steps
            writer.add_scalar(f"loss_step/{split_name}", step_loss, global_step)
        if total_steps % max(1, LOG_TRAIN_PROGRESS_EVERY) == 0:
            logging.info(
                "%s epoch=%s step=%s/%s loss=%.6f",
                split_name,
                epoch,
                total_steps,
                len(loader),
                step_loss,
            )
        loss_parts["trade"] += float(loss_trade.detach().cpu().item())
        loss_parts["dir"] += float(loss_dir.detach().cpu().item())
        loss_parts["bounce"] += float(loss_bounce.detach().cpu().item())
        loss_parts["ksl"] += float(loss_ksl.detach().cpu().item())
        loss_parts["ktp"] += float(loss_ktp.detach().cpu().item())
        loss_parts["ttp"] += float(loss_ttp.detach().cpu().item())
        loss_parts["edge"] += float(loss_edge.detach().cpu().item())
        loss_parts["qdist"] += float(loss_qdist.detach().cpu().item())
        loss_parts["trend"] += float(loss_trend.detach().cpu().item())
        loss_parts["trend_dir"] += float(loss_trend_dir.detach().cpu().item())
        loss_parts["transition"] += float(loss_transition.detach().cpu().item())
        loss_parts["combiner"] += float(loss_combiner.detach().cpu().item())
        loss_parts["dist_exit"] += float(loss_dist.detach().cpu().item())
        loss_parts["break"] += float(loss_break.detach().cpu().item())

        logits["trade"].append(trade_logit.detach().cpu().numpy())
        logits["dir"].append(dir_logit.detach().cpu().numpy())
        logits["bounce"].append(bounce_logit.detach().cpu().numpy())
        logits["ttp_quick"].append(ttp_logit.detach().cpu().numpy())
        logits["edge"].append(edge_pred.detach().cpu().numpy())
        logits["qdist"].append(qdist_pred.detach().cpu().numpy())
        logits["trend"].append(p_trend_gate.detach().cpu().numpy())
        logits["trend_up"].append(p_trend_up.detach().cpu().numpy())
        logits["transition"].append(transition_logit.detach().cpu().numpy())
        if USE_LEARNED_ENTRY_COMBINER:
            logits["combiner"].append(combiner_logit.detach().cpu().numpy())
        else:
            logits["combiner"].append(np.zeros_like(y_trade.detach().cpu().numpy()))
        logits["ksl"].append(ksl_pred.detach().cpu().numpy())
        logits["ktp"].append(ktp_pred.detach().cpu().numpy())
        logits["dist_exit"].append(dist_pred.detach().cpu().numpy())
        logits["break"].append(break_pred.detach().cpu().numpy())

        labels["y_trade"].append(y_trade.detach().cpu().numpy())
        labels["y_dir"].append(y_dir.detach().cpu().numpy())
        labels["y_bounce"].append(y_bounce.detach().cpu().numpy())
        labels["y_ttp_quick"].append(y_ttp_quick.detach().cpu().numpy())
        labels["y_edge"].append(y_edge.detach().cpu().numpy())
        labels["y_qdist"].append(y_qdist.detach().cpu().numpy())
        labels["y_trend"].append(y_trend.detach().cpu().numpy())
        labels["y_trend_up"].append(y_trend_up.detach().cpu().numpy())
        labels["y_trend_up_soft"].append(y_trend_up_soft.detach().cpu().numpy())
        labels["y_transition"].append(y_transition.detach().cpu().numpy())
        if trend_strength_norm is not None:
            labels["trend_strength_norm"].append(trend_strength_norm.detach().cpu().numpy())
        else:
            labels["trend_strength_norm"].append(np.zeros_like(y_trend.detach().cpu().numpy()))
        if trend_adv is not None:
            labels["trend_adv"].append(trend_adv.detach().cpu().numpy())
        else:
            labels["trend_adv"].append(np.zeros_like(y_trend.detach().cpu().numpy()))
        labels["mask_dir"].append(mask_dir.detach().cpu().numpy())
        labels["mask_bounce"].append(mask_bounce.detach().cpu().numpy())
        labels["mask_ttp"].append(mask_ttp.detach().cpu().numpy())
        labels["mask_edge"].append(mask_edge.detach().cpu().numpy())
        labels["mask_qdist"].append(mask_qdist.detach().cpu().numpy())
        labels["mask_trend"].append(mask_trend.detach().cpu().numpy())
        labels["mask_transition"].append(mask_transition.detach().cpu().numpy())
        if USE_LEARNED_ENTRY_COMBINER:
            labels["y_combiner"].append(y_eff.detach().cpu().numpy())
            labels["mask_combiner"].append(mask_comb.detach().cpu().numpy())
        else:
            labels["y_combiner"].append(np.zeros_like(y_trade.detach().cpu().numpy()))
            labels["mask_combiner"].append(np.zeros_like(y_trade.detach().cpu().numpy()))
        labels["y_ksl"].append(y_ksl.detach().cpu().numpy())
        labels["mask_ksl"].append(mask_ksl.detach().cpu().numpy())
        labels["y_ktp"].append(y_ktp.detach().cpu().numpy())
        labels["mask_ktp"].append(mask_ktp.detach().cpu().numpy())
        labels["y_dist_targets"].append(y_dist_targets.detach().cpu().numpy())
        labels["mask_dist"].append(mask_dist.detach().cpu().numpy())
        labels["y_break"].append(y_break.detach().cpu().numpy())
        labels["mask_break"].append(mask_break.detach().cpu().numpy())
        labels["excess_up_atr"].append(excess_up_atr.detach().cpu().numpy())
        labels["excess_dn_atr"].append(excess_dn_atr.detach().cpu().numpy())
        bar_ts.extend(batch["bar_ts"].numpy())

    for key in logits:
        logits[key] = np.concatenate(logits[key], axis=0)
    for key in labels:
        labels[key] = np.concatenate(labels[key], axis=0)

    for key in loss_parts:
        loss_parts[key] = loss_parts[key] / max(1, total_steps)
    grad_norm_mean = float(np.mean(grad_norms)) if grad_norms else 0.0

    pred_arr = np.array(diag["trend_dir_preds"], dtype=np.float32)
    label_arr = np.array(diag["trend_dir_labels"], dtype=np.float32)
    mask_arr = np.ones_like(pred_arr, dtype=bool)
    diag_stats = _masked_stats_probs(pred_arr, mask_arr)
    diag_pos_rate = float(diag["trend_dir_pos_sum"] / max(1, diag["trend_dir_count"]))
    diag_acc = float(diag["trend_dir_acc_sum"] / max(1, diag["trend_dir_samples"])) if diag["trend_dir_samples"] else 0.0
    diag_brier = float(diag["trend_dir_brier_sum"] / max(1, diag["trend_dir_samples"])) if diag["trend_dir_samples"] else 0.0
    diag_auc = _masked_auc(pred_arr, label_arr, mask_arr) if pred_arr.size else float("nan")
    diag_grad = diag["trend_dir_grad_norm"] / max(1, diag["trend_dir_grad_steps"])

    return total_loss / max(1, total_steps), {
        "logits": logits,
        "labels": labels,
        "bar_ts": np.array(bar_ts),
        "loss_parts": loss_parts,
        "grad_norm": grad_norm_mean,
        "diag": {
            "trend_dir_count": float(diag["trend_dir_count"]),
            "trend_dir_pos_rate": diag_pos_rate,
            "trend_dir_acc": diag_acc,
            "trend_dir_brier": diag_brier,
            "trend_dir_auc": diag_auc,
            "trend_dir_empty_batches": float(diag["trend_dir_empty_batches"]),
            "trend_dir_grad_norm": float(diag_grad),
            **diag_stats,
        },
    }


def _log_epoch_metrics(
    writer: object,
    split: str,
    epoch: int,
    loss: float,
    metrics: dict[str, float],
    logits: dict[str, np.ndarray],
    feature_stats: dict[str, tuple[float, float]],
) -> None:
    writer.add_scalar(f"loss/{split}_total", loss, epoch)
    writer.add_scalar(f"metrics/{split}_trade_acc", metrics["trade_acc"], epoch)
    writer.add_scalar(f"metrics/{split}_dir_acc_on_trades", metrics["dir_acc_on_trades"], epoch)
    writer.add_scalar(f"metrics/{split}_bounce_acc_on_near", metrics["bounce_acc_on_near"], epoch)
    writer.add_scalar(f"metrics/{split}_bounce_precision_on_near", metrics["bounce_precision_on_near"], epoch)
    writer.add_scalar(f"metrics/{split}_bounce_recall_on_near", metrics["bounce_recall_on_near"], epoch)
    writer.add_scalar(f"metrics/{split}_trend_acc", metrics["trend_acc"], epoch)
    writer.add_scalar(f"metrics/{split}_trend_dir_acc", metrics["trend_dir_acc"], epoch)
    writer.add_scalar(f"metrics/{split}_trend_dir_corr_adv", metrics["trend_dir_corr_adv"], epoch)
    writer.add_scalar(f"metrics/{split}_trend_dir_brier_w", metrics["trend_dir_brier_w"], epoch)
    writer.add_scalar(f"metrics/{split}_trend_gate_pos_mean", metrics["trend_gate_pos_mean"], epoch)
    writer.add_scalar(f"metrics/{split}_trend_gate_neg_mean", metrics["trend_gate_neg_mean"], epoch)
    writer.add_scalar(f"metrics/{split}_coverage", metrics["coverage"], epoch)
    writer.add_scalar(f"metrics/{split}_trades_per_day", metrics["trades_per_day"], epoch)
    writer.add_scalar(f"labels/{split}_trend_rate", metrics["trend_rate"], epoch)
    writer.add_scalar(f"labels/{split}_opportunity_rate", metrics["opportunity_rate"], epoch)
    writer.add_scalar(f"labels/{split}_near_level_rate", metrics["near_level_rate"], epoch)
    writer.add_scalar(f"labels/{split}_bounce_mask_rate", metrics["bounce_mask_rate"], epoch)
    writer.add_scalar(f"labels/{split}_bounce_rate", metrics["bounce_rate"], epoch)
    if split == "val":
        writer.add_scalar("breakout/val_break_acc_on_mask", metrics["break_acc_on_mask"], epoch)
        writer.add_scalar("breakout/val_break_precision_on_mask", metrics["break_precision_on_mask"], epoch)
        writer.add_scalar("breakout/val_break_recall_on_mask", metrics["break_recall_on_mask"], epoch)
        writer.add_scalar("breakout/val_break_mask_rate", metrics["break_mask_rate"], epoch)
        for key, val in metrics.items():
            if key.startswith("break_success_rate_bin_"):
                writer.add_scalar(f"breakout/val_{key}", val, epoch)
            if key.startswith("break_p_break_mean_bin_"):
                writer.add_scalar(f"breakout/val_{key}", val, epoch)
    if split == "val":
        writer.add_scalar("sltp/val_ksl_mae_atr", metrics["ksl_mae"], epoch)
        writer.add_scalar("sltp/val_ksl_rmse_atr", metrics["ksl_rmse"], epoch)
        writer.add_scalar("sltp/val_ksl_corr", metrics["ksl_corr"], epoch)
        writer.add_scalar("sltp/val_ksl_coverage", metrics["ksl_coverage"], epoch)
        writer.add_scalar("sltp/val_ktp_mae_atr", metrics["ktp_mae"], epoch)
        writer.add_scalar("sltp/val_ktp_rmse_atr", metrics["ktp_rmse"], epoch)
        writer.add_scalar("sltp/val_ktp_corr", metrics["ktp_corr"], epoch)
        writer.add_scalar("sltp/val_ktp_coverage", metrics["ktp_coverage"], epoch)
        writer.add_scalar("sltp/val_y_ksl_mean", metrics["y_ksl_mean"], epoch)
        writer.add_scalar("sltp/val_y_ksl_p90", metrics["y_ksl_p90"], epoch)
        writer.add_scalar("sltp/val_y_ktp_mean", metrics["y_ktp_mean"], epoch)
        writer.add_scalar("sltp/val_y_ktp_p90", metrics["y_ktp_p90"], epoch)
        writer.add_scalar("ttp/val_loss", metrics["ttp_loss"], epoch)
        writer.add_scalar("ttp/val_acc", metrics["ttp_acc"], epoch)
        writer.add_scalar("ttp/val_rate_pos", metrics["ttp_rate_pos"], epoch)
        writer.add_scalar("ttp/val_pred_mean", metrics["ttp_pred_mean"], epoch)
        writer.add_scalar("edge/val_mae_atr", metrics["edge_mae"], epoch)
        writer.add_scalar("edge/val_neg_rate", metrics["edge_neg_rate"], epoch)
        writer.add_scalar("edge/val_neg_mean", metrics["edge_neg_mean"], epoch)
        writer.add_scalar("qdist/val_pinball", metrics["qdist_pinball"], epoch)
        if USE_TRANSITION_HEAD:
            writer.add_scalar("transition/val_loss", metrics["transition_loss"], epoch)
            writer.add_scalar("transition/val_acc", metrics["transition_acc"], epoch)
            writer.add_scalar("transition/val_rate_pos", metrics["transition_rate_pos"], epoch)
            writer.add_scalar("transition/val_pred_mean", metrics["transition_pred_mean"], epoch)
        writer.add_scalar("dist_exit/val_mae_q50_h5", metrics["dist_mae_q50_h5"], epoch)
        writer.add_scalar("dist_exit/val_pinball_mean", metrics["dist_pinball_mean"], epoch)

    for thr in (0.55, 0.60, 0.70):
        writer.add_scalar(
            f"metrics/{split}_selective_dir_acc_{thr}", metrics[f"selective_dir_acc_{thr}"], epoch
        )

    writer.add_scalar(f"head2/{split}_p_trade_mean", metrics["p_trade_mean"], epoch)
    writer.add_scalar(f"head2/{split}_p_up_mean", metrics["p_up_mean"], epoch)
    writer.add_scalar(f"head3/{split}_p_bounce_mean", metrics["p_bounce_mean"], epoch)
    writer.add_scalar(f"head_ttp/{split}_p_ttp_mean", metrics["p_ttp_mean"], epoch)
    writer.add_scalar(f"trend_gate/{split}_p_trend_gate_mean", metrics["p_trend_gate_mean"], epoch)
    writer.add_scalar(f"trend_gate/{split}_p_trend_up_mean", metrics["p_trend_up_mean"], epoch)
    writer.add_scalar(f"trend_gate/{split}_pos_mean", metrics["trend_gate_pos_mean"], epoch)
    writer.add_scalar(f"trend_gate/{split}_neg_mean", metrics["trend_gate_neg_mean"], epoch)
    writer.add_histogram(f"head2/{split}_p_trade_hist", torch.tensor(1 / (1 + np.exp(-logits["trade"]))), epoch)
    writer.add_histogram(f"head2/{split}_p_up_hist", torch.tensor(1 / (1 + np.exp(-logits["dir"]))), epoch)
    writer.add_histogram(f"head3/{split}_p_bounce_hist", torch.tensor(1 / (1 + np.exp(-logits["bounce"]))), epoch)
    writer.add_histogram(
        f"head_ttp/{split}_p_ttp_hist", torch.tensor(1 / (1 + np.exp(-logits["ttp_quick"]))), epoch
    )
    for name, (mn, mx) in feature_stats.items():
        writer.add_scalar(f"feature_stats/{name}_min", mn, epoch)
        writer.add_scalar(f"feature_stats/{name}_max", mx, epoch)


def _train_and_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: Optional[dict] = None,
    effective_config: Optional[dict] = None,
) -> None:
    _ensure_output_dirs()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=RUNS_DIR / f"trend_mr_{run_id}")
    cfg_path = dump_run_config_json(Path(writer.log_dir))
    logging.info("Run config written: %s", cfg_path)
    if config is not None:
        writer.add_text("config/json", json.dumps(config, indent=2, default=str))
    if effective_config is not None:
        writer.add_text("config/effective", json.dumps(effective_config, indent=2, default=str))
    writer.add_scalar("config/use_slow_stream", 1.0 if USE_SLOW_STREAM else 0.0, 0)
    writer.add_scalar("config/run_smoke_test", 1.0 if RUN_SMOKE_TEST else 0.0, 0)

    logging.info("TREND-GATE FIX: using raw close windows separate from model features.")
    train_end, _ = _split_train_val(len(df))
    labels = _compute_labels(df, train_end)

    _resolve_trend_indices(feature_cols)

    norm_features, norm_params = _normalize_features(df, feature_cols, train_end)
    if "xau_close" not in df.columns:
        raise KeyError("xau_close missing from dataframe; trend gate requires raw close series.")
    close_prices = df["xau_close"].to_numpy(dtype=np.float32)
    if USE_SLOW_STREAM:
        slow_features, slow_map = _build_slow_features(df, feature_cols)
    else:
        slow_features, slow_map = None, None

    bar_ts = (
        pd.to_datetime(df["bar_dt"], utc=True, errors="coerce")
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )
    dataset = SequenceDataset(
        norm_features,
        close_prices,
        slow_features,
        slow_map,
        labels,
        bar_ts,
        USE_SLOW_STREAM,
        len(feature_cols),
    )
    if any(idx is not None for idx in (TRAIN_START_IDX, TRAIN_END_IDX, VAL_START_IDX, VAL_END_IDX)):
        train_start = int(TRAIN_START_IDX) if isinstance(TRAIN_START_IDX, int) and TRAIN_START_IDX >= 0 else 0
        train_end_idx = int(train_end)
        val_start = int(VAL_START_IDX) if isinstance(VAL_START_IDX, int) and VAL_START_IDX >= 0 else train_end_idx
        val_end = int(VAL_END_IDX) if isinstance(VAL_END_IDX, int) and VAL_END_IDX > 0 else len(df)
        train_idx = list(range(train_start, max(train_start, train_end_idx)))
        val_idx = list(range(val_start, max(val_start, val_end)))
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
    else:
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_end, len(df) - train_end])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = TrendMRModel(in_features=len(feature_cols), slow_in_features=len(feature_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    feature_stats = _feature_stats(df)
    diag_enabled = bool(DIAG_EXPORT)
    diag_split = str(DIAG_EXPORT_SPLIT).lower()
    train_diag = None
    val_diag = None
    if diag_enabled:
        if diag_split in ("train", "both"):
            train_diag = _DiagCollector(DIAG_EXPORT_MAX_ROWS, DIAG_EXPORT_FORMAT, DIAG_EXPORT_DIR, "train")
        if diag_split in ("val", "both"):
            val_diag = _DiagCollector(DIAG_EXPORT_MAX_ROWS, DIAG_EXPORT_FORMAT, DIAG_EXPORT_DIR, "val")

    for epoch in range(1, EPOCHS + 1):
        logging.info("Starting epoch %s/%s", epoch, EPOCHS)
        if diag_enabled and train_diag is not None:
            train_diag.rows = []
            train_diag.total_seen = 0
        if diag_enabled and val_diag is not None:
            val_diag.rows = []
            val_diag.total_seen = 0

        train_loss, train_payload = _run_epoch(
            model, train_loader, optimizer, "train", epoch, writer=writer, diag_collector=train_diag
        )
        val_loss, val_payload = _run_epoch(
            model, val_loader, None, "val", epoch, writer=writer, diag_collector=val_diag
        )

        train_metrics = _compute_metrics(train_payload["logits"], train_payload["labels"], train_payload["bar_ts"])
        val_metrics = _compute_metrics(val_payload["logits"], val_payload["labels"], val_payload["bar_ts"])

        _log_epoch_metrics(writer, "train", epoch, train_loss, train_metrics, train_payload["logits"], feature_stats)
        _log_epoch_metrics(writer, "val", epoch, val_loss, val_metrics, val_payload["logits"], feature_stats)

        # Trend-dir diagnostics interpretation:
        # - low count/high empty_batches => noisy validation
        # - skewed pos_rate => class imbalance
        # - high p01/p99 => saturation/overconfidence
        # - near-zero grad_norm => head not learning
        for split_name, payload in (("train", train_payload), ("val", val_payload)):
            diag = payload.get("diag", {})
            if diag:
                writer.add_scalar(f"diag/{split_name}_trend_dir_count", diag.get("trend_dir_count", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_pos_rate", diag.get("trend_dir_pos_rate", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_acc", diag.get("trend_dir_acc", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_auc", diag.get("trend_dir_auc", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_brier", diag.get("trend_dir_brier", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p_mean", diag.get("mean", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p_min", diag.get("min", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p_max", diag.get("max", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p01_frac", diag.get("p01_frac", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p99_frac", diag.get("p99_frac", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p1", diag.get("p1", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p5", diag.get("p5", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p10", diag.get("p10", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p25", diag.get("p25", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p50", diag.get("p50", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p75", diag.get("p75", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p90", diag.get("p90", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p95", diag.get("p95", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_p99", diag.get("p99", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_empty_batches", diag.get("trend_dir_empty_batches", 0.0), epoch)
                writer.add_scalar(f"diag/{split_name}_trend_dir_grad_norm", diag.get("trend_dir_grad_norm", 0.0), epoch)

        train_parts = train_payload.get("loss_parts", {})
        val_parts = val_payload.get("loss_parts", {})
        writer.add_scalar("loss/train_trade_gate", float(train_parts.get("trade", 0.0)), epoch)
        writer.add_scalar("loss/train_direction", float(train_parts.get("dir", 0.0)), epoch)
        writer.add_scalar("loss/train_bounce", float(train_parts.get("bounce", 0.0)), epoch)
        writer.add_scalar("loss/train_ksl", float(train_parts.get("ksl", 0.0)), epoch)
        writer.add_scalar("loss/train_ktp", float(train_parts.get("ktp", 0.0)), epoch)
        writer.add_scalar("loss/train_ttp", float(train_parts.get("ttp", 0.0)), epoch)
        writer.add_scalar("loss/train_edge", float(train_parts.get("edge", 0.0)), epoch)
        writer.add_scalar("loss/train_qdist", float(train_parts.get("qdist", 0.0)), epoch)
        writer.add_scalar("loss/train_trend", float(train_parts.get("trend", 0.0)), epoch)
        writer.add_scalar("loss/train_trend_dir", float(train_parts.get("trend_dir", 0.0)), epoch)
        writer.add_scalar("loss/train_transition", float(train_parts.get("transition", 0.0)), epoch)
        writer.add_scalar("loss/train_combiner", float(train_parts.get("combiner", 0.0)), epoch)
        writer.add_scalar("loss/train_dist_exit", float(train_parts.get("dist_exit", 0.0)), epoch)
        writer.add_scalar("loss/train_break", float(train_parts.get("break", 0.0)), epoch)
        writer.add_scalar("loss/val_trade_gate", float(val_parts.get("trade", 0.0)), epoch)
        writer.add_scalar("loss/val_direction", float(val_parts.get("dir", 0.0)), epoch)
        writer.add_scalar("loss/val_bounce", float(val_parts.get("bounce", 0.0)), epoch)
        writer.add_scalar("loss/val_ksl", float(val_parts.get("ksl", 0.0)), epoch)
        writer.add_scalar("loss/val_ktp", float(val_parts.get("ktp", 0.0)), epoch)
        writer.add_scalar("loss/val_ttp", float(val_parts.get("ttp", 0.0)), epoch)
        writer.add_scalar("loss/val_edge", float(val_parts.get("edge", 0.0)), epoch)
        writer.add_scalar("loss/val_qdist", float(val_parts.get("qdist", 0.0)), epoch)
        writer.add_scalar("loss/val_trend", float(val_parts.get("trend", 0.0)), epoch)
        writer.add_scalar("loss/val_trend_dir", float(val_parts.get("trend_dir", 0.0)), epoch)
        writer.add_scalar("loss/val_transition", float(val_parts.get("transition", 0.0)), epoch)
        writer.add_scalar("loss/val_combiner", float(val_parts.get("combiner", 0.0)), epoch)
        writer.add_scalar("loss/val_dist_exit", float(val_parts.get("dist_exit", 0.0)), epoch)
        writer.add_scalar("loss/val_break", float(val_parts.get("break", 0.0)), epoch)

        writer.add_scalar("debug/grad_norm", float(train_payload.get("grad_norm", 0.0)), epoch)
        writer.add_scalar("debug/lr", float(optimizer.param_groups[0]["lr"]), epoch)
        nan_inf_count = int(np.size(df[feature_cols].to_numpy()) - np.isfinite(df[feature_cols].to_numpy()).sum())
        writer.add_scalar("debug/feature_nan_inf_count", nan_inf_count, epoch)

        _append_metrics(
            METRICS_DIR / "train_metrics.csv",
            {
                "run_id": run_id,
                "epoch": epoch,
                "loss": train_loss,
                **train_metrics,
            },
        )
        _append_metrics(
            METRICS_DIR / "val_metrics.csv",
            {
                "run_id": run_id,
                "epoch": epoch,
                "loss": val_loss,
                **val_metrics,
            },
        )

        if diag_enabled and (epoch % max(1, int(DIAG_EXPORT_EVERY)) == 0):
            if train_diag is not None:
                path = train_diag.write(epoch)
                if path is not None:
                    logging.info("DIAG_EXPORT: wrote %s rows to %s", len(train_diag.rows), path)
            if val_diag is not None:
                path = val_diag.write(epoch)
                if path is not None:
                    logging.info("DIAG_EXPORT: wrote %s rows to %s", len(val_diag.rows), path)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "norm_params": norm_params,
            "feature_cols": feature_cols,
        }
        torch.save(ckpt, CHECKPOINT_DIR / f"trend_mr_epoch{epoch}.pt")

        logging.info(
            "Epoch %s done: train_loss=%.4f val_loss=%.4f coverage=%.3f",
            epoch,
            train_loss,
            val_loss,
            val_metrics["coverage"],
        )

    writer.flush()
    writer.close()


def _smoke_test(df: pd.DataFrame, feature_cols: list[str]) -> None:
    logging.info("Smoke test: training on small slice.")
    small = df.iloc[:2000].copy()
    _train_and_validate(small, feature_cols)


def _latest_checkpoint_path() -> Path:
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {CHECKPOINT_DIR}")
    candidates = list(CHECKPOINT_DIR.glob("trend_mr_epoch*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _fetch_m1(symbol: str, n_bars: int) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 module unavailable; cannot fetch live bars.")
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, n_bars)
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["bar_dt", "open", "high", "low", "close", "volume", "spread"])
    df = pd.DataFrame(rates)
    df["bar_dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["open"] = df["open"]
    df["high"] = df["high"]
    df["low"] = df["low"]
    df["close"] = df["close"]
    df["volume"] = df.get("tick_volume", df.get("real_volume", 0.0))
    df["spread"] = df.get("spread", 0.0)
    df = df[["bar_dt", "open", "high", "low", "close", "volume", "spread"]]
    return _normalize_bar_dt(df)


def _round_lot_step(x: float, step: float = 0.01) -> float:
    if step <= 0:
        return x
    return round(round(x / step) * step, 2)


def _compute_lot(p_trade: float, p_up: float, direction: int) -> float:
    dir_conf = p_up if direction > 0 else (1.0 - p_up)
    score = max(0.0, min(1.0, p_trade * dir_conf))
    lot = LOT_MIN + score * (LOT_MAX - LOT_MIN)
    lot = _round_lot_step(lot, 0.01)
    return min(LOT_MAX, max(LOT_MIN, lot))


def _compute_kelly_lot(
    p_trade_eff: float,
    q10_hat: float,
    q50_hat: float,
    q90_hat: float,
) -> tuple[float, float]:
    eps = 1e-8
    p_win = float(p_trade_eff) if KELLY_USE_P_TRADE_EFF else 1.0
    if KELLY_MODE == "dist_quantiles":
        denom = max(-q10_hat, eps)
        f_raw = max(0.0, q50_hat / denom)
    else:
        avg_win = max(q90_hat, 0.0)
        avg_loss = max(-q10_hat, eps)
        b = avg_win / avg_loss
        f_raw = p_win - (1.0 - p_win) / max(b, eps)
    f = _clamp(f_raw, KELLY_CLIP_MIN, KELLY_CLIP_MAX)
    f *= float(KELLY_FRACTIONAL)
    if KELLY_USE_TAIL_RISK:
        tail_risk = max(-q10_hat, 0.0)
        f *= math.exp(-float(KELLY_TAIL_RISK_LAMBDA) * tail_risk)
    f = _clamp(f, 0.0, 1.0)
    lot = float(LOT_MIN) + f * (float(LOT_MAX) - float(LOT_MIN))
    lot = _round_lot_step(lot, float(KELLY_LOT_STEP))
    lot = min(float(LOT_MAX), max(float(LOT_MIN), lot))
    return lot, f


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _clip_if(x: float, lo: float, hi: float, enabled: bool) -> float:
    return max(lo, min(float(x), hi)) if enabled else float(x)


def _compute_sl_distance_px_for_labels(atr_px_t: float, k_sl_t: float) -> float:
    if atr_px_t <= 0.0:
        return 0.0
    k_eff = float(k_sl_t)
    if APPLY_K_CLAMPS:
        k_eff = _clip_if(k_eff, K_SL_MIN, K_SL_MAX, True)
    sl_dist = atr_px_t * float(SL_ATR_MULT) * k_eff
    if SL_EXTRA_DOLLARS > 0.0:
        extra_px = float(SL_EXTRA_DOLLARS)
        if MAX_SL_EXTRA_ATR is not None:
            extra_px = min(extra_px, float(MAX_SL_EXTRA_ATR) * atr_px_t)
        sl_dist += extra_px
    return max(sl_dist, 0.0)


def _compute_trailing_sl_px(
    side: int,
    sl_px: float,
    entry_px: float,
    best_px: float,
    close_px: float,
    atr_px: float,
    trail_start_atr: float,
    trail_atr_mult: float,
) -> tuple[float, float]:
    if atr_px <= 0.0:
        return sl_px, best_px
    if side > 0:
        best_px = max(best_px, close_px)
        move = best_px - entry_px
        if move < trail_start_atr * atr_px:
            return sl_px, best_px
        proposed = best_px - trail_atr_mult * atr_px
        new_sl = max(sl_px, proposed)
        if new_sl - sl_px < TRAIL_UPDATE_MIN_STEP_ATR * atr_px:
            return sl_px, best_px
        return new_sl, best_px
    best_px = min(best_px, close_px)
    move = entry_px - best_px
    if move < trail_start_atr * atr_px:
        return sl_px, best_px
    proposed = best_px + trail_atr_mult * atr_px
    new_sl = min(sl_px, proposed)
    if sl_px - new_sl < TRAIL_UPDATE_MIN_STEP_ATR * atr_px:
        return sl_px, best_px
    return new_sl, best_px


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _dir_confidence(p_up: float, direction: int) -> float:
    return float(p_up) if direction > 0 else float(1.0 - p_up)


def _logit_from_prob(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(p / (1.0 - p))


def _combiner_input_dim() -> int:
    dim = 4  # trade, trend, trend_dir, base direction
    if COMBINER_INCLUDE_TTP:
        dim += 1
    if COMBINER_INCLUDE_EDGE:
        dim += 1
    if COMBINER_INCLUDE_QDIST:
        dim += len(QDIST_QUANTILES)
    if COMBINER_INCLUDE_TAIL:
        dim += 1
    if COMBINER_INCLUDE_CONTEXT:
        dim += 3  # vol, spread, trend_strength_norm
    return dim


def _transition_input_dim() -> int:
    dim = 0
    if TRANSITION_INCLUDE_SHARED:
        dim += EMBED_DIM
    if TRANSITION_INCLUDE_SLOPES:
        dim += len(SLOPE_WINDOWS)
    if TRANSITION_INCLUDE_DISAGREE:
        dim += 10
    return dim


def _slope_ols_1d_torch(price: torch.Tensor) -> torch.Tensor:
    n = price.shape[-1]
    if n <= 1:
        return torch.zeros(price.shape[:-1], device=price.device, dtype=price.dtype)
    t = torch.arange(n, device=price.device, dtype=price.dtype)
    t = t - t.mean()
    y = price - price.mean(dim=-1, keepdim=True)
    denom = torch.sum(t * t)
    if denom <= 1e-12:
        return torch.zeros(price.shape[:-1], device=price.device, dtype=price.dtype)
    return torch.sum(y * t, dim=-1) / denom


def _compute_slopes_multi_torch(price_seq: torch.Tensor, windows: Iterable[int]) -> torch.Tensor:
    slopes = []
    seq_len = price_seq.shape[-2]
    for w in windows:
        w = int(w)
        w = max(2, min(w, seq_len))
        window = price_seq[:, -w:, 0]
        slopes.append(_slope_ols_1d_torch(window))
    return torch.stack(slopes, dim=-1)


def _compute_disagree_features_torch(slopes_norm: torch.Tensor) -> torch.Tensor:
    k = slopes_norm.shape[-1]
    short_idx = 0
    long_idx = k - 1
    mid_idx = _nearest_index(SLOPE_WINDOWS, 32)
    large_idx = _nearest_index(SLOPE_WINDOWS, 64)
    slope_short = slopes_norm[:, short_idx]
    slope_long = slopes_norm[:, long_idx]
    slope_mid = slopes_norm[:, mid_idx]
    slope_large = slopes_norm[:, large_idx]
    abs_short = torch.abs(slope_short)
    abs_long = torch.abs(slope_long)
    abs_mid = torch.abs(slope_mid)
    min_strength = float(TRANSITION_MIN_STRENGTH)
    sign_short = torch.sign(slope_short)
    sign_long = torch.sign(slope_long)
    sign_mid = torch.sign(slope_mid)
    sign_large = torch.sign(slope_large)
    sign_mismatch_sl = ((sign_short * sign_long) < 0).float() * (
        (abs_short > min_strength) & (abs_long > min_strength)
    ).float()
    sign_mismatch_sm = ((sign_short * sign_mid) < 0).float() * (
        (abs_short > min_strength) & (abs_mid > min_strength)
    ).float()
    sign_mismatch_ml = ((sign_mid * sign_long) < 0).float() * (
        (abs_mid > min_strength) & (abs_long > min_strength)
    ).float()
    gap_sl = slope_short - slope_long
    gap_ml = slope_mid - slope_long
    same_dir = torch.sum(torch.sign(slopes_norm) == torch.sign(slope_long.unsqueeze(-1)), dim=-1).float()
    opposite_dir = float(k) - same_dir
    disp = torch.std(slopes_norm, dim=-1)
    feats = [
        sign_mismatch_sl,
        sign_mismatch_sm,
        sign_mismatch_ml,
        gap_sl,
        gap_ml,
        torch.abs(gap_sl),
        torch.abs(gap_ml),
        same_dir,
        opposite_dir,
        disp,
    ]
    return torch.stack(feats, dim=-1)


def _slope_ols_1d_np(price: np.ndarray) -> float:
    n = price.shape[0]
    if n <= 1:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t = t - t.mean()
    y = price - price.mean()
    denom = np.sum(t * t)
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(y * t) / denom)


def _compute_slopes_multi_np(price: np.ndarray, windows: Iterable[int]) -> np.ndarray:
    slopes = []
    seq_len = price.shape[0]
    for w in windows:
        w = int(w)
        w = max(2, min(w, seq_len))
        window = price[-w:]
        slopes.append(_slope_ols_1d_np(window))
    return np.array(slopes, dtype=np.float32)


def _transition_dir_from_seq(
    close_short_seq: np.ndarray,
    trend_short_seq: Optional[np.ndarray] = None,
) -> tuple[int, bool, np.ndarray]:
    if close_short_seq.size == 0:
        return 0, False, np.zeros((len(SLOPE_WINDOWS),), dtype=np.float32)
    price_series = close_short_seq.astype(np.float64)
    slopes = _compute_slopes_multi_np(price_series, SLOPE_WINDOWS)
    if SLOPE_USE_ATR_NORM:
        if trend_short_seq is not None and TREND_VOL_COL_IDX is not None:
            atr_px = float(trend_short_seq[-1, TREND_VOL_COL_IDX])
        else:
            atr_px = float(abs(price_series[-1] - price_series[-2])) if price_series.size > 1 else 0.0
        atr_px = max(atr_px, SLOPE_ATR_EPS)
        slopes = slopes / atr_px
    slopes = np.clip(slopes, -DISAGREE_CLIP, DISAGREE_CLIP)
    if slopes.size == 0:
        return 0, False, slopes
    slope_short = float(slopes[0])
    slope_long = float(slopes[-1])
    if abs(slope_short) < TRANSITION_MIN_STRENGTH or abs(slope_long) < TRANSITION_MIN_STRENGTH:
        return 0, False, slopes
    disagree = np.sign(slope_short) != np.sign(slope_long)
    if not disagree:
        return 0, False, slopes
    if TRANSITION_DIR_SOURCE == "vote7":
        valid = np.abs(slopes) >= TRANSITION_MIN_STRENGTH
        if not np.any(valid):
            return 0, False, slopes
        sign_sum = float(np.sum(np.sign(slopes[valid])))
        if sign_sum == 0.0:
            return 0, False, slopes
        direction = 1 if sign_sum > 0 else -1
    else:
        direction = 1 if slope_short > 0 else -1
    return direction, True, slopes


def _safe_percentile(values: list[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _clip_thr(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(np.clip(x, CALIBRATION_THR_FLOOR, CALIBRATION_THR_CEIL))


def _nearest_index(values: Iterable[int], target: int) -> int:
    vals = list(values)
    if not vals:
        return 0
    return int(min(range(len(vals)), key=lambda i: abs(vals[i] - target)))


def _quantile_index(target: int) -> int:
    return _nearest_index(DIST_EXIT_QUANTILES, target)


def _horizon_index(target: int) -> int:
    return _nearest_index(DIST_EXIT_HORIZONS, target)


def _qdist_index(target: float) -> int:
    vals = list(QDIST_QUANTILES)
    if not vals:
        return 0
    return int(min(range(len(vals)), key=lambda i: abs(vals[i] - target)))


def _dynamic_atr_mults(
    p_trade: float,
    p_up: float,
    p_trade_std: float,
    p_up_std: float,
    direction: int,
) -> tuple[float, float, float, float]:
    dir_conf = _clamp(_dir_confidence(p_up, direction), 0.0, 1.0)
    score = _clamp(float(p_trade) * dir_conf, 0.0, 1.0)

    unc = max(0.0, float(p_trade_std) + float(p_up_std))
    unc_norm = _clamp(unc / max(UNCERTAINTY_NORM, 1e-8), 0.0, 1.0)

    k_sl_base = _lerp(SL_ATR_MULT_MIN, SL_ATR_MULT_MAX, score)
    k_tp_base = _lerp(TP_ATR_MULT_MIN, TP_ATR_MULT_MAX, score)

    penalty = UNCERTAINTY_PENALTY * unc_norm
    k_sl = _clamp(k_sl_base * (1.0 - 0.35 * penalty), SL_ATR_MULT_MIN, SL_ATR_MULT_MAX)
    k_tp = _clamp(k_tp_base * (1.0 - 0.50 * penalty), TP_ATR_MULT_MIN, TP_ATR_MULT_MAX)

    if DYNAMIC_TRAIL_MULTS:
        trail_atr_mult = _clamp(
            _lerp(TRAIL_ATR_MULT_MIN, TRAIL_ATR_MULT_MAX, score) * (1.0 - 0.40 * penalty),
            TRAIL_ATR_MULT_MIN,
            TRAIL_ATR_MULT_MAX,
        )
        trail_start_atr = _clamp(
            _lerp(TRAIL_START_ATR_MIN, TRAIL_START_ATR_MAX, 1.0 - score) * (1.0 + 0.30 * penalty),
            TRAIL_START_ATR_MIN,
            TRAIL_START_ATR_MAX,
        )
    else:
        trail_atr_mult = float(TRAIL_ATR_MULT)
        trail_start_atr = float(TRAIL_START_ATR)

    return float(k_sl), float(k_tp), float(trail_start_atr), float(trail_atr_mult)


def _trend_diag_from_seq(
    close_short_seq: np.ndarray,
    close_long_seq: np.ndarray,
    trend_short_seq: Optional[np.ndarray] = None,
) -> dict[str, float]:
    if close_short_seq.size == 0:
        return {}
    price_short = close_short_seq.astype(np.float64)
    price_long = close_long_seq.astype(np.float64)

    if TREND_LONG_POOL_K > 1 and price_long.size >= TREND_LONG_POOL_K:
        steps = price_long.size // TREND_LONG_POOL_K
        price_long = price_long[-steps * TREND_LONG_POOL_K :]
        price_long = price_long.reshape(steps, TREND_LONG_POOL_K)
        if TREND_LONG_POOL_MODE == "last":
            price_long = price_long[:, -1]
        else:
            price_long = price_long.mean(axis=1)

    def _slope(arr: np.ndarray) -> float:
        if arr.size < 2:
            return 0.0
        t = np.arange(arr.size, dtype=np.float64)
        t = t - t.mean()
        x = arr - np.mean(arr)
        cov = np.sum(x * t)
        var = np.sum(t * t) + TREND_EPS
        return float(cov / var)

    slope_short = _slope(price_short)
    slope_long = _slope(price_long)
    price_last = float(price_short[-1])
    if trend_short_seq is not None and TREND_VOL_COL_IDX is not None:
        vol_last = float(trend_short_seq[-1, TREND_VOL_COL_IDX])
        atr_px = abs(vol_last * price_last) + TREND_EPS
    else:
        atr_px = abs(price_short[-1] - price_short[-2]) + TREND_EPS if price_short.size > 1 else TREND_EPS

    slope_norm = (slope_short + slope_long) * 0.5 / atr_px
    w = min(TREND_SHORT_PUSH_WINDOW, max(1, price_short.size - 1))
    delta = np.diff(price_short)
    delta_win = delta[-w:]
    push_up_atr = float(np.max(np.maximum(delta_win, 0.0)) / atr_px) if delta_win.size else 0.0
    push_dn_atr = float(np.max(np.maximum(-delta_win, 0.0)) / atr_px) if delta_win.size else 0.0
    price_win = price_short[-w:]
    peak = np.maximum.accumulate(price_win)
    trough = np.minimum.accumulate(price_win)
    pullback_dn_atr = float(np.max(peak - price_win) / atr_px) if price_win.size else 0.0
    pullback_up_atr = float(np.max(price_win - trough) / atr_px) if price_win.size else 0.0
    ratio_up = float(push_up_atr / (pullback_dn_atr + TREND_EPS))
    ratio_dn = float(push_dn_atr / (pullback_up_atr + TREND_EPS))

    return {
        "slope_norm": slope_norm,
        "push_up_atr": push_up_atr,
        "push_dn_atr": push_dn_atr,
        "pullback_dn_atr": pullback_dn_atr,
        "pullback_up_atr": pullback_up_atr,
        "ratio_up": ratio_up,
        "ratio_dn": ratio_dn,
    }


def _true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def _atr(df: pd.DataFrame, period: int) -> float:
    if df.empty or period <= 0:
        return 0.0
    if len(df) < 3:
        return 0.0
    df_closed = df.iloc[:-1]
    if len(df_closed) < 2:
        return 0.0
    highs = df_closed["high"].to_numpy(dtype=np.float64)
    lows = df_closed["low"].to_numpy(dtype=np.float64)
    closes = df_closed["close"].to_numpy(dtype=np.float64)
    start_idx = max(1, len(df_closed) - (period + 1))
    trs = []
    for i in range(start_idx, len(df_closed)):
        tr = _true_range(float(highs[i]), float(lows[i]), float(closes[i - 1]))
        trs.append(tr)
    if not trs:
        return 0.0
    tr_series = pd.Series(trs, dtype="float64")
    if len(tr_series) >= period:
        atr_val = tr_series.rolling(period).mean().iloc[-1]
    else:
        atr_val = tr_series.mean()
    if not np.isfinite(atr_val):
        return 0.0
    return float(atr_val)


def _build_live_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    norm_params: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    norm_matrix = _apply_norm_params(df, feature_cols, norm_params)
    idx = len(df) - 1
    if "xau_close" not in df.columns:
        raise KeyError("xau_close missing from live dataframe; trend gate requires raw close series.")
    closes = df["xau_close"].to_numpy(dtype=np.float64)
    if closes.size == 0:
        raise ValueError("xau_close is empty; trend gate cannot build close windows.")
    short_seq = _build_sequence(norm_matrix, idx, SHORT_LEN)
    mid_seq = _build_sequence(norm_matrix, idx, MID_LEN)
    long_seq = _build_sequence(norm_matrix, idx, LONG_LEN)
    trend_short_seq = _build_sequence(norm_matrix, idx, TREND_SHORT_LOOKBACK_BARS)
    trend_long_seq = _build_sequence(norm_matrix, idx, TREND_LONG_LOOKBACK_BARS)
    close_short_seq = _build_close_sequence(closes, idx, TREND_SHORT_LOOKBACK_BARS)
    close_long_seq = _build_close_sequence(closes, idx, TREND_LONG_LOOKBACK_BARS)
    if USE_SLOW_STREAM:
        slow_seq = _build_slow_sequence_live(df, feature_cols)
    else:
        slow_seq = np.zeros((SLOW_LEN, len(feature_cols)), dtype=np.float32)
    return short_seq, mid_seq, long_seq, slow_seq, trend_short_seq, trend_long_seq, close_short_seq, close_long_seq, idx


def _beta_from_window(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    var = float(np.var(x))
    if var <= 0:
        return float("nan")
    cov = float(np.cov(x, y, ddof=0)[0, 1])
    return cov / var


def _fast_update_live_features(df: pd.DataFrame, xau: pd.DataFrame, xag: pd.DataFrame, dxy: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    last_idx = df.index[-1]
    if not xau.empty:
        df.loc[last_idx, "xau_close"] = float(xau["close"].iloc[-1])
        df.loc[last_idx, "xau_spread"] = float(xau["spread"].iloc[-1]) * SPREAD_POINTS_TO_PRICE
    if "xag_close" in df.columns and not xag.empty:
        df.loc[last_idx, "xag_close"] = float(xag["close"].iloc[-1])
    if "dxy_close" in df.columns and not dxy.empty:
        df.loc[last_idx, "dxy_close"] = float(dxy["close"].iloc[-1])

    closes = df["xau_close"].to_numpy(dtype=np.float64)
    if closes.size < 2:
        return df
    prev_close = closes[-2]
    curr_close = closes[-1]
    df.loc[last_idx, "ret_1m"] = (curr_close / prev_close - 1.0) if prev_close > 0 else 0.0
    for w in RET_WINDOWS:
        if int(w) == 1:
            continue
        if len(closes) > w and closes[-(w + 1)] > 0:
            df.loc[last_idx, f"ret_{w}m"] = curr_close / closes[-(w + 1)] - 1.0
        else:
            df.loc[last_idx, f"ret_{w}m"] = 0.0
    ret_1m = df["ret_1m"].to_numpy(dtype=np.float64)
    window = ret_1m[-VOL_WINDOW:] if ret_1m.size >= 1 else np.array([0.0])
    start_timer("volatility")
    df.loc[last_idx, "vol_30m"] = float(np.std(window)) if window.size else 0.0
    stop_timer("volatility")

    xag_close = df["xag_close"].to_numpy(dtype=np.float64) if "xag_close" in df.columns else None
    dxy_close = df["dxy_close"].to_numpy(dtype=np.float64) if "dxy_close" in df.columns else None
    ret_xag_last = 0.0
    ret_dxy_last = 0.0
    if xag_close is not None and xag_close.size >= 2 and xag_close[-2] > 0:
        ret_xag_last = xag_close[-1] / xag_close[-2] - 1.0
    if dxy_close is not None and dxy_close.size >= 2 and dxy_close[-2] > 0:
        ret_dxy_last = dxy_close[-1] / dxy_close[-2] - 1.0
    if "sign_agree_xag" in df.columns:
        df.loc[last_idx, "sign_agree_xag"] = float(np.sign(ret_1m[-1]) == np.sign(ret_xag_last))
    if "sign_agree_dxy" in df.columns:
        df.loc[last_idx, "sign_agree_dxy"] = float(np.sign(ret_1m[-1]) == -np.sign(ret_dxy_last))

    start_timer("correlations")
    for w in CORR_WINDOWS:
        w = int(w)
        if w <= 1 or ret_1m.size < w:
            continue
        idx_slice = slice(-w, None)
        ret_1m_win = ret_1m[idx_slice]
        if f"corr_xau_xag_{w}" in df.columns and xag_close is not None and xag_close.size >= w + 1:
            ret_xag_win = xag_close[-w:] / xag_close[-(w + 1):-1] - 1.0
            corr = _safe_corrcoef(ret_1m_win, ret_xag_win, f"xau_xag_{w}")
            df.loc[last_idx, f"corr_xau_xag_{w}"] = corr if np.isfinite(corr) else 0.0
        if f"corr_xau_dxy_{w}" in df.columns and dxy_close is not None and dxy_close.size >= w + 1:
            ret_dxy_win = dxy_close[-w:] / dxy_close[-(w + 1):-1] - 1.0
            corr = _safe_corrcoef(ret_1m_win, ret_dxy_win, f"xau_dxy_{w}")
            df.loc[last_idx, f"corr_xau_dxy_{w}"] = corr if np.isfinite(corr) else 0.0
        if f"beta_xag_to_xau_{w}" in df.columns and xag_close is not None and xag_close.size >= w + 1:
            ret_xag_win = xag_close[-w:] / xag_close[-(w + 1):-1] - 1.0
            beta = _beta_from_window(ret_xag_win, ret_1m_win)
            df.loc[last_idx, f"beta_xag_to_xau_{w}"] = 0.0 if not np.isfinite(beta) else float(beta)
        if f"beta_xau_to_dxy_{w}" in df.columns and dxy_close is not None and dxy_close.size >= w + 1:
            ret_dxy_win = dxy_close[-w:] / dxy_close[-(w + 1):-1] - 1.0
            beta = _beta_from_window(ret_1m_win, ret_dxy_win)
            df.loc[last_idx, f"beta_xau_to_dxy_{w}"] = 0.0 if not np.isfinite(beta) else float(beta)
        if f"xaucore_{w}" in df.columns:
            beta = df.loc[last_idx, f"beta_xau_to_dxy_{w}"] if f"beta_xau_to_dxy_{w}" in df.columns else 0.0
            df.loc[last_idx, f"xaucore_{w}"] = float(ret_1m[-1] + beta * ret_dxy_last)
    stop_timer("correlations")

    if USE_SESSION_FLAGS:
        sess = _session_flags(df.loc[last_idx, "bar_dt"])
        df.loc[last_idx, ["session_asia", "session_london", "session_ny", "london_open", "ny_open", "minute_sin", "minute_cos"]] = sess
        df.loc[last_idx, "session_id"] = 0 if sess[0] == 1 else 1 if sess[1] == 1 else 2
        vol_med_window = SESSION_MEDIAN_DAYS * 1440
        sess_id = int(df.loc[last_idx, "session_id"])
        mask = df["session_id"] == sess_id
        median = float(df.loc[mask, "vol_30m"].tail(vol_med_window).median()) if mask.any() else 0.0
        df.loc[last_idx, "vol_session_median_5d"] = median
        df.loc[last_idx, "vol_session_ratio"] = float(df.loc[last_idx, "vol_30m"]) / (median + 1e-8)

    return df


def _calibrate_entry_thresholds_on_startup(
    df: pd.DataFrame,
    feature_cols: list[str],
    norm_params: dict[str, tuple[float, float]],
    model: TrendMRModel,
    *,
    use_full_dataset: bool,
) -> dict[str, float]:
    needed = max(SHORT_LEN, MID_LEN, LONG_LEN, TREND_SHORT_LOOKBACK_BARS, TREND_LONG_LOOKBACK_BARS, SLOW_LEN)
    if df.empty or len(df) < needed:
        logging.warning("[CALIBRATION] Not enough rows (%s) for lookback=%s", len(df), needed)
        return {}
    norm_matrix = _apply_norm_params(df, feature_cols, norm_params)
    end_idx = len(df) - 1
    if use_full_dataset:
        start_idx = needed - 1
    else:
        start_idx = max(needed - 1, end_idx - CALIBRATION_BARS + 1)
    if start_idx >= end_idx:
        logging.warning("[CALIBRATION] Insufficient calibration range.")
        return {}

    if "xau_close" not in df.columns:
        raise KeyError("xau_close missing from dataframe; trend gate requires raw close series.")
    close_prices = df["xau_close"].to_numpy(dtype=np.float32)
    slow_seq = (
        _build_slow_sequence_live(df, feature_cols) if USE_SLOW_STREAM else np.zeros((SLOW_LEN, len(feature_cols)), dtype=np.float32)
    )

    trend_scores: list[float] = []
    mr_scores: list[float] = []
    gate_source = "combiner"
    batch_size = 128
    idxs = list(range(start_idx, end_idx + 1))
    total_batches = int(math.ceil(len(idxs) / batch_size)) if idxs else 0
    logging.info("[CALIBRATION] scoring batches=%s (rows=%s, batch_size=%s)", total_batches, len(idxs), batch_size)
    t0 = time.time()

    with torch.no_grad():
        for b_start in range(0, len(idxs), batch_size):
            batch_idx = idxs[b_start : b_start + batch_size]
            batch_id = b_start // batch_size + 1
            if batch_id == 1 or batch_id % 50 == 0:
                elapsed = time.time() - t0
                logging.info(
                    "[CALIBRATION] batch %s/%s (elapsed=%.1fs)",
                    batch_id,
                    total_batches,
                    elapsed,
                )
            short_list = []
            mid_list = []
            long_list = []
            trend_short_list = []
            trend_long_list = []
            close_short_list = []
            close_long_list = []
            for idx in batch_idx:
                short_list.append(_build_sequence(norm_matrix, idx, SHORT_LEN))
                mid_list.append(_build_sequence(norm_matrix, idx, MID_LEN))
                long_list.append(_build_sequence(norm_matrix, idx, LONG_LEN))
                trend_short_list.append(_build_sequence(norm_matrix, idx, TREND_SHORT_LOOKBACK_BARS))
                trend_long_list.append(_build_sequence(norm_matrix, idx, TREND_LONG_LOOKBACK_BARS))
                close_short_list.append(_build_close_sequence(close_prices, idx, TREND_SHORT_LOOKBACK_BARS))
                close_long_list.append(_build_close_sequence(close_prices, idx, TREND_LONG_LOOKBACK_BARS))

            short = torch.from_numpy(np.stack(short_list)).to(DEVICE)
            mid = torch.from_numpy(np.stack(mid_list)).to(DEVICE)
            long = torch.from_numpy(np.stack(long_list)).to(DEVICE)
            slow = torch.from_numpy(np.repeat(slow_seq[None, :, :], len(batch_idx), axis=0)).to(DEVICE)
            trend_short = torch.from_numpy(np.stack(trend_short_list)).to(DEVICE)
            trend_long = torch.from_numpy(np.stack(trend_long_list)).to(DEVICE)
            close_short = torch.from_numpy(np.stack(close_short_list)).to(DEVICE)
            close_long = torch.from_numpy(np.stack(close_long_list)).to(DEVICE)

            (
                trade_logit,
                dir_logit,
                _,
                ttp_logit,
                edge_pred,
                qdist_pred,
                p_trend_gate,
                _,
                _,
                _,
                _,
                p_trend_up,
                _,
            ) = model(short, mid, long, slow, trend_short, trend_long, close_short, close_long)

            if USE_LEARNED_ENTRY_COMBINER and hasattr(model, "entry_combiner"):
                trade_feat = trade_logit if COMBINER_USE_LOGITS else torch.sigmoid(trade_logit)
                trend_feat = _logit_from_prob(p_trend_gate) if COMBINER_USE_LOGITS else p_trend_gate
                trend_dir_feat = _logit_from_prob(p_trend_up) if COMBINER_USE_LOGITS else p_trend_up
                dir_feat = dir_logit if COMBINER_USE_LOGITS else torch.sigmoid(dir_logit)
                feats = [trade_feat, trend_feat, trend_dir_feat, dir_feat]
                if COMBINER_INCLUDE_TTP:
                    ttp_feat = ttp_logit if COMBINER_USE_LOGITS else torch.sigmoid(ttp_logit)
                    ttp_feat = torch.clamp(ttp_feat, -COMBINER_TTP_CLIP, COMBINER_TTP_CLIP)
                    feats.append(ttp_feat)
                if COMBINER_INCLUDE_EDGE:
                    edge_feat = edge_pred
                    edge_feat = torch.clamp(edge_feat, -COMBINER_EDGE_CLIP, COMBINER_EDGE_CLIP)
                    feats.append(edge_feat)
                if COMBINER_INCLUDE_QDIST:
                    qdist_feat = torch.clamp(qdist_pred, -COMBINER_QDIST_CLIP, COMBINER_QDIST_CLIP)
                    feats.extend([qdist_feat[:, i] for i in range(qdist_feat.shape[1])])
                if COMBINER_INCLUDE_TAIL:
                    tail_asym = qdist_pred[:, -1] - qdist_pred[:, 0]
                    tail_asym = torch.clamp(tail_asym, -COMBINER_TAIL_CLIP, COMBINER_TAIL_CLIP)
                    feats.append(tail_asym)
                if COMBINER_INCLUDE_CONTEXT:
                    vol_vals = df.loc[batch_idx, "vol_30m"].to_numpy(dtype=np.float32) if "vol_30m" in df.columns else np.zeros(len(batch_idx), dtype=np.float32)
                    spread_vals = df.loc[batch_idx, "xau_spread"].to_numpy(dtype=np.float32) if "xau_spread" in df.columns else np.zeros(len(batch_idx), dtype=np.float32)
                    strength_vals = np.zeros(len(batch_idx), dtype=np.float32)
                    feats.extend(
                        [
                            torch.from_numpy(vol_vals).to(DEVICE),
                            torch.from_numpy(spread_vals).to(DEVICE),
                            torch.from_numpy(strength_vals).to(DEVICE),
                        ]
                    )
                comb_in = torch.stack(feats, dim=-1)
                combiner_logit = model.entry_combiner(comb_in)
                p_trade_eff = torch.sigmoid(combiner_logit).detach().cpu().numpy()
            else:
                gate_source = "p_trade_only"
                p_trade_eff = torch.sigmoid(trade_logit).detach().cpu().numpy()

            p_trend_vals = p_trend_gate.detach().cpu().numpy()
            for score, is_trend in zip(p_trade_eff, p_trend_vals):
                if np.isfinite(is_trend):
                    if is_trend >= 0.5:
                        trend_scores.append(float(score))
                    else:
                        mr_scores.append(float(score))
                else:
                    if CALIBRATION_DEFAULT_REGIME.upper() == "TREND":
                        trend_scores.append(float(score))
                    else:
                        mr_scores.append(float(score))

    stats = {}
    trend_p50 = _safe_percentile(trend_scores, 50)
    trend_p75 = _safe_percentile(trend_scores, CALIBRATION_PERCENTILE)
    trend_p90 = _safe_percentile(trend_scores, 90)
    trend_p95 = _safe_percentile(trend_scores, 95)
    trend_p99 = _safe_percentile(trend_scores, 99)
    trend_mean = float(np.mean(trend_scores)) if trend_scores else float("nan")
    trend_min = float(np.min(trend_scores)) if trend_scores else float("nan")
    trend_max = float(np.max(trend_scores)) if trend_scores else float("nan")
    mr_p50 = _safe_percentile(mr_scores, 50)
    mr_p75 = _safe_percentile(mr_scores, CALIBRATION_PERCENTILE)
    mr_p90 = _safe_percentile(mr_scores, 90)
    mr_p95 = _safe_percentile(mr_scores, 95)
    mr_p99 = _safe_percentile(mr_scores, 99)
    mr_mean = float(np.mean(mr_scores)) if mr_scores else float("nan")
    mr_min = float(np.min(mr_scores)) if mr_scores else float("nan")
    mr_max = float(np.max(mr_scores)) if mr_scores else float("nan")

    stats.update(
        {
            "trend_count": len(trend_scores),
            "mr_count": len(mr_scores),
            "trend_p50": trend_p50,
            "trend_p75": trend_p75,
            "trend_p90": trend_p90,
            "trend_p95": trend_p95,
            "trend_p99": trend_p99,
            "trend_mean": trend_mean,
            "trend_min": trend_min,
            "trend_max": trend_max,
            "mr_p50": mr_p50,
            "mr_p75": mr_p75,
            "mr_p90": mr_p90,
            "mr_p95": mr_p95,
            "mr_p99": mr_p99,
            "mr_mean": mr_mean,
            "mr_min": mr_min,
            "mr_max": mr_max,
            "gate_source": gate_source,
        }
    )

    return stats


@dataclass
class LiveConfig:
    symbol: str
    lots: float
    slippage: int
    magic: int
    comment: str
    timeframe: int


class TrendMRLiveTrader:
    def __init__(
        self,
        model: TrendMRModel,
        feature_cols: list[str],
        norm_params: dict[str, tuple[float, float]],
        usd_symbol: str,
        dry_run: bool,
        exit_confirm_bars: int,
    ) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.norm_params = norm_params
        self.usd_symbol = usd_symbol
        self.dry_run = dry_run
        self.exit_confirm_bars = max(1, int(exit_confirm_bars))
        self.cfg = LiveConfig(
            symbol=TARGET_SYMBOL,
            lots=LIVE_LOTS,
            slippage=LIVE_SLIPPAGE,
            magic=LIVE_MAGIC,
            comment=LIVE_COMMENT,
            timeframe=mt5.TIMEFRAME_M1 if mt5 is not None else 0,
        )
        self.position: Optional[dict] = None
        self.last_seen_bar_ts: Optional[pd.Timestamp] = None
        self.cached_df: Optional[pd.DataFrame] = None
        self.cached_feature_cols: Optional[list[str]] = None
        self.cached_full_ts: Optional[pd.Timestamp] = None
        self.log_path = OUTPUT_DIR / LIVE_LOG_FILENAME
        self.last_p_trade_std = 0.0
        self.last_p_up_std = 0.0
        self.last_atr = 0.0
        self.last_k_sl = float(SL_ATR_MULT)
        self.last_k_tp = float(TP_ATR_MULT)
        self.last_k_sl_raw = float(SL_ATR_MULT)
        self.last_k_tp_raw = float(TP_ATR_MULT)
        self.last_k_sl_used = float(SL_ATR_MULT)
        self.last_k_tp_used = float(TP_ATR_MULT)
        self.last_k_sl_before_extra = float(SL_ATR_MULT)
        self.last_k_sl_final = float(SL_ATR_MULT)
        self.last_sl_extra_atr = 0.0
        self.last_dist_ev_atr = 0.0
        self.last_dist_q10_atr = 0.0
        self.last_dist_q50_atr = 0.0
        self.last_dist_q75_atr = 0.0
        self.last_dist_q95_atr = 0.0
        self.last_dist_exit_cond = 0
        self.last_trail_start_atr = float(TRAIL_START_ATR)
        self.last_trail_atr_mult = float(TRAIL_ATR_MULT)
        self.bar_index = 0
        self.last_stopout_bar_idx: Optional[int] = None
        self.last_stopout_direction: Optional[int] = None
        self.trend_mode = False
        self.trend_enter_count = 0
        self.trend_exit_count = 0
        self.trend_switches = 0
        self.last_trend_log_bar = -1
        self.last_transition_log_bar = -1
        self.transition_exit_streak = 0
        self.eff_buf_trend = deque(maxlen=int(EFF_THR_ROLL_N))
        self.eff_buf_mr = deque(maxlen=int(EFF_THR_ROLL_N))
        self.eff_thr_trend = float(EFF_THR_STATIC_TREND)
        self.eff_thr_mr = float(EFF_THR_STATIC_MR)
        self.last_thr_update_bar = -1
        self.timing_writer: Optional[SummaryWriter] = None
        self.timing_step = 0
        if ENABLE_LIVE_FEATURE_TIMING:
            try:
                self.timing_writer = SummaryWriter(RUNS_DIR / "live_timing")
            except Exception:
                self.timing_writer = None
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        "bar_ts",
                        "close",
                        "spread",
                        "p_trade",
                        "p_entry_ttp_quick",
                        "p_trade_eff",
                        "edge_hat_atr",
                        "q10_hat",
                        "q50_hat",
                        "q90_hat",
                        "kelly_f",
                        "kelly_lot",
                        "eff_thr",
                        "p_trade_std",
                        "p_up",
                        "p_up_std",
                        "p_bounce",
                        "p_trend",
                        "p_transition",
                        "transition_dir",
                        "dist_ev_atr",
                        "dist_q10_atr",
                        "dist_q50_atr",
                        "dist_q75_atr",
                        "dist_q95_atr",
                        "dist_exit_cond",
                        "p_break",
                        "excess_up_atr",
                        "excess_dn_atr",
                        "atr",
                        "k_sl_raw",
                        "k_tp_raw",
                        "k_sl_used",
                        "k_tp_used",
                        "k_sl_before_extra",
                        "sl_extra_dollars",
                        "sl_extra_atr",
                        "k_sl_final",
                        "trail_start_atr",
                        "trail_atr_mult",
                        "sl",
                        "tp",
                        "lot",
                        "sl_price",
                        "tp_price",
                        "action",
                        "entry_side",
                        "exit_now",
                        "exit_streak",
                        "exit_reason",
                        "hold_bars",
                    ]
                )

    def _ensure_connection(self) -> None:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 module unavailable; install/enable MT5 to run live trading.")
        if not _mt5_initialize():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
        term = mt5.terminal_info()
        term_path = getattr(term, "path", None) if term is not None else None
        acct = mt5.account_info()
        acct_login = getattr(acct, "login", None) if acct is not None else None
        logging.info("MT5 terminal path: %s", term_path)
        logging.info("MT5 account login: %s", acct_login)
        for symbol in (self.cfg.symbol, XAG_SYMBOL, self.usd_symbol):
            info = mt5.symbol_info(symbol)
            if info is None or (not info.visible and not mt5.symbol_select(symbol, True)):
                raise RuntimeError(f"Symbol {symbol} unavailable in MT5.")
        logging.info("Connected to MT5 build %s", mt5.version())
        info = _sym_info_safe(self.cfg.symbol)
        logging.info("Symbol constraints: %s", info)

    def _get_mt5_position(self) -> Optional[object]:
        if mt5 is None:
            return None
        sym = self.cfg.symbol
        poss = mt5.positions_get(symbol=sym)
        if not poss:
            return None
        magic = getattr(self.cfg, "magic", None)
        if magic is not None:
            poss2 = [p for p in poss if getattr(p, "magic", None) == magic]
        else:
            poss2 = list(poss)
        if not poss2:
            return None
        local_ticket = int(self.position.get("ticket", 0)) if self.position else 0
        if local_ticket:
            for p in poss2:
                if int(getattr(p, "ticket", 0)) == local_ticket:
                    return p

        def _pos_key(p: object) -> tuple[int, int, int]:
            return (
                int(getattr(p, "time_msc", 0) or 0),
                int(getattr(p, "time", 0) or 0),
                int(getattr(p, "ticket", 0) or 0),
            )

        poss2.sort(key=_pos_key, reverse=True)
        return poss2[0]

    def _pos_side(self, pos: object) -> str:
        return "BUY" if getattr(pos, "type", None) == mt5.POSITION_TYPE_BUY else "SELL"

    def _update_regime_state(self, p_trend_gate: float) -> bool:
        if REGIME_BLEND_MODE:
            self.trend_mode = p_trend_gate >= 0.5
            return self.trend_mode
        if not self.trend_mode:
            if p_trend_gate >= TREND_ENTER_THR:
                self.trend_enter_count += 1
            else:
                self.trend_enter_count = 0
            if self.trend_enter_count >= TREND_PERSIST_BARS:
                self.trend_mode = True
                self.trend_enter_count = 0
                self.trend_exit_count = 0
                self.trend_switches += 1
                logging.info("Regime switch: TREND (p_trend=%.3f)", p_trend_gate)
        else:
            if p_trend_gate <= TREND_EXIT_THR:
                self.trend_exit_count += 1
            else:
                self.trend_exit_count = 0
            if self.trend_exit_count >= TREND_PERSIST_BARS:
                self.trend_mode = False
                self.trend_exit_count = 0
                self.trend_enter_count = 0
                self.trend_switches += 1
                logging.info("Regime switch: MEAN_REVERSION (p_trend=%.3f)", p_trend_gate)
        return self.trend_mode

    def _normalize_volume(self, symbol: str, volume: float) -> float:
        info = mt5.symbol_info(symbol)
        if info is None:
            return volume
        step = getattr(info, "volume_step", 0.01) or 0.01
        step = max(0.01, float(step))
        min_vol = getattr(info, "volume_min", step) or step
        min_vol = max(0.01, float(min_vol))
        max_vol = getattr(info, "volume_max", volume) or volume
        vol = max(min_vol, min(volume, max_vol))
        vol = round(vol / step) * step
        return max(min_vol, min(vol, max_vol))

    def _get_last_price(self) -> Optional[float]:
        tick = mt5.symbol_info_tick(self.cfg.symbol)
        if tick:
            price = tick.ask or tick.last or tick.bid
            if price:
                return float(price)
        rates = mt5.copy_rates_from_pos(self.cfg.symbol, self.cfg.timeframe, 0, 1)
        if rates is None or len(rates) == 0:
            return None
        return float(rates[0]["close"])

    def _get_live_spread(self) -> float:
        tick = mt5.symbol_info_tick(self.cfg.symbol)
        if tick and tick.ask and tick.bid:
            return float(tick.ask - tick.bid)
        return 0.0

    def _open_position(
        self,
        direction: int,
        price: float,
        atr_points: float,
        lot_override: Optional[float] = None,
        sl_mult: Optional[float] = None,
        tp_mult: Optional[float] = None,
    ) -> bool:
        if self.position is not None:
            logging.debug("Position already open; skip opening a new one.")
            return False
        base_lot = self.cfg.lots if lot_override is None else lot_override
        lot = self._normalize_volume(self.cfg.symbol, base_lot)
        sl_price = None
        tp_price = None
        eff_sl_mult = float(sl_mult) if sl_mult is not None else float(SL_ATR_MULT)
        eff_tp_mult = float(tp_mult) if tp_mult is not None else float(TP_ATR_MULT)
        if atr_points > 0.0:
            if ENABLE_SL:
                sl_price = price - direction * eff_sl_mult * atr_points
            if ENABLE_TP:
                tp_price = price + direction * eff_tp_mult * atr_points
        if self.dry_run:
            self.position = {
                "ticket": 0,
                "direction": direction,
                "volume": lot,
                "entry_price": price,
                "hold_bars": 0,
                "exit_streak": 0,
                "sl": sl_price,
                "tp": tp_price,
                "sl_mult": eff_sl_mult,
                "tp_mult": eff_tp_mult,
                "best_price": price,
                "atr_at_entry": atr_points,
            }
            logging.info("DRY RUN: opened %s at %.2f", "long" if direction > 0 else "short", price)
            return True
        info = _sym_info_safe(self.cfg.symbol)
        vol_min = float(info.get("volume_min") or 0.0)
        vol_max = float(info.get("volume_max") or lot)
        vol_step = float(info.get("volume_step") or 0.01)
        rounded_lot = _round_to_step(lot, vol_step)
        rounded_lot = min(max(rounded_lot, vol_min), vol_max if vol_max > 0 else rounded_lot)
        if vol_min and rounded_lot < vol_min:
            logging.error(
                "Abort open: volume < min (requested=%.4f rounded=%.4f min=%.4f step=%.4f max=%.4f)",
                lot,
                rounded_lot,
                vol_min,
                vol_step,
                vol_max,
            )
            return False
        lot = rounded_lot
        tick = _get_tick(self.cfg.symbol)
        if tick is None:
            logging.error("Abort open: tick unavailable for %s", self.cfg.symbol)
            return False
        bid = float(tick.bid) if tick.bid else price
        ask = float(tick.ask) if tick.ask else price
        side = "BUY" if direction > 0 else "SELL"
        exec_price = ask if side == "BUY" else bid
        if exec_price <= 0:
            logging.error("Abort open: invalid exec price bid=%.5f ask=%.5f", bid, ask)
            return False
        sl_dist = abs(price - sl_price) if sl_price is not None else 0.0
        tp_dist = abs(tp_price - price) if tp_price is not None else 0.0
        logging.info(
            "SLTP debug(entry): side=%s entry=%.5f bid=%.5f ask=%.5f atr=%.6f k_sl_raw=%.4f k_tp_raw=%.4f "
            "k_sl_used=%.4f k_tp_used=%.4f sl_dist=%.5f tp_dist=%.5f sl=%.5f tp=%.5f "
            "enable_sl=%s enable_tp=%s stops_level=%s freeze_level=%s vol_min=%.4f vol_step=%.4f vol_max=%.4f",
            side,
            exec_price,
            bid,
            ask,
            atr_points,
            float(self.last_k_sl_raw),
            float(self.last_k_tp_raw),
            eff_sl_mult,
            eff_tp_mult,
            sl_dist,
            tp_dist,
            sl_price if sl_price is not None else 0.0,
            tp_price if tp_price is not None else 0.0,
            ENABLE_SL,
            ENABLE_TP,
            info.get("trade_stops_level"),
            info.get("trade_freeze_level"),
            vol_min,
            vol_step,
            vol_max,
        )
        sl_ok, tp_ok, sl_reason, tp_reason = _validate_sltp(
            side,
            exec_price,
            bid,
            ask,
            sl_price,
            tp_price,
            info.get("trade_stops_level"),
            info.get("trade_freeze_level"),
            info.get("point"),
        )
        if ENABLE_SL and sl_price is not None and not sl_ok:
            logging.warning("Disabled SL: %s", sl_reason)
            sl_price = None
        if ENABLE_TP and tp_price is not None and not tp_ok:
            logging.warning("Disabled TP: %s", tp_reason)
            tp_price = None
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL,
            "price": exec_price,
            "deviation": self.cfg.slippage,
            "comment": self.cfg.comment,
            "magic": self.cfg.magic,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if ENABLE_SL and sl_price is not None:
            request["sl"] = sl_price
        if ENABLE_TP and tp_price is not None:
            request["tp"] = tp_price
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(
                "Failed to open trade: retcode=%s comment=%s last_error=%s side=%s sl=%.5f tp=%.5f bid=%.5f ask=%.5f",
                getattr(result, "retcode", None),
                getattr(result, "comment", None),
                mt5.last_error() if mt5 is not None else None,
                side,
                sl_price or 0.0,
                tp_price or 0.0,
                bid,
                ask,
            )
            return False
        self.position = {
            "ticket": 0,
            "direction": direction,
            "volume": lot,
            "entry_price": price,
            "hold_bars": 0,
            "exit_streak": 0,
            "sl": sl_price,
            "tp": tp_price,
            "sl_mult": eff_sl_mult,
            "tp_mult": eff_tp_mult,
            "best_price": price,
            "atr_at_entry": atr_points,
        }
        time.sleep(0.2)
        pos = self._get_mt5_position()
        if pos is not None:
            self.position["ticket"] = int(getattr(pos, "ticket", 0))
            self.position["direction"] = 1 if self._pos_side(pos) == "BUY" else -1
            self.position["entry_price"] = float(getattr(pos, "price_open", price))
        else:
            logging.warning("Order filled but position not found; will resync next bar.")
        logging.info("Opened %s at %.2f", "long" if direction > 0 else "short", price)
        return True

    def _modify_position_sltp(self, new_sl: Optional[float], new_tp: Optional[float] = None) -> bool:
        if self.position is None:
            return False
        if self.dry_run:
            self.position["sl"] = new_sl
            return True
        pos = self._get_mt5_position()
        if pos is None:
            logging.warning("No MT5 position found; skipping modify and clearing local state.")
            self.position = None
            return False
        info = _sym_info_safe(self.cfg.symbol)
        tick = mt5.symbol_info_tick(self.cfg.symbol)
        bid = float(tick.bid) if tick and tick.bid else 0.0
        ask = float(tick.ask) if tick and tick.ask else 0.0
        local_dir = int(self.position.get("direction", 0)) if self.position else 0
        side = self._pos_side(pos)
        if (local_dir > 0 and side != "BUY") or (local_dir < 0 and side != "SELL"):
            logging.warning("Local direction != MT5 position type; using MT5 as source of truth.")
        ticket = int(getattr(pos, "ticket", 0))
        if ticket == 0:
            logging.error("Abort modify: MT5 position ticket is 0.")
            return False
        tp_val = new_tp if new_tp is not None else self.position.get("tp")
        sl_ok, tp_ok, sl_reason, tp_reason = _validate_sltp(
            side,
            float(getattr(pos, "price_open", 0.0)),
            bid,
            ask,
            new_sl,
            tp_val,
            info.get("trade_stops_level"),
            info.get("trade_freeze_level"),
            info.get("point"),
        )
        if ENABLE_SL and new_sl is not None and not sl_ok:
            logging.warning("Disabled SL: %s", sl_reason)
            new_sl = None
        if ENABLE_TP and tp_val is not None and not tp_ok:
            logging.warning("Disabled TP: %s", tp_reason)
            tp_val = None
        logging.info(
            "SLTP debug(modify): side=%s bid=%.5f ask=%.5f sl=%.5f tp=%.5f stops_level=%s freeze_level=%s",
            side,
            bid,
            ask,
            new_sl if new_sl is not None else 0.0,
            tp_val if tp_val is not None else 0.0,
            info.get("trade_stops_level"),
            info.get("trade_freeze_level"),
        )
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": self.cfg.symbol,
            "sl": float(new_sl) if ENABLE_SL and new_sl is not None else 0.0,
            "tp": float(tp_val) if ENABLE_TP and tp_val is not None else 0.0,
            "magic": self.cfg.magic,
            "comment": self.cfg.comment,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(
                "Failed to modify SL/TP: retcode=%s comment=%s last_error=%s ticket=%s side=%s sl=%.5f tp=%.5f bid=%.5f ask=%.5f",
                getattr(result, "retcode", None),
                getattr(result, "comment", None),
                mt5.last_error() if mt5 is not None else None,
                ticket,
                side,
                new_sl or 0.0,
                tp_val or 0.0,
                bid,
                ask,
            )
            return False
        self.position["sl"] = new_sl
        return True

    def _update_trailing_stop(
        self,
        atr_current: float,
        bid: Optional[float],
        ask: Optional[float],
        close: float,
        trail_start_atr: float,
        trail_atr_mult: float,
    ) -> None:
        if self.position is None or not ENABLE_TRAILING or not ENABLE_SL:
            return
        if atr_current <= 0.0:
            return
        pos = self._get_mt5_position()
        if pos is None:
            logging.warning("No MT5 position found during trailing; skipping update.")
            return
        side = self._pos_side(pos)
        entry_price = float(self.position.get("entry_price", close))
        best_price = float(self.position.get("best_price", entry_price))
        if side == "BUY":
            ref_price = bid if bid is not None and bid > 0 else close
            best_price = max(best_price, ref_price)
            move = best_price - entry_price
            if move < trail_start_atr * atr_current:
                self.position["best_price"] = best_price
                return
            proposed_sl = best_price - trail_atr_mult * atr_current
            current_sl = self.position.get("sl")
            new_sl = proposed_sl if current_sl is None else max(float(current_sl), proposed_sl)
            improve = current_sl is None or (new_sl - float(current_sl) >= TRAIL_UPDATE_MIN_STEP_ATR * atr_current)
        else:
            ref_price = ask if ask is not None and ask > 0 else close
            best_price = min(best_price, ref_price)
            move = entry_price - best_price
            if move < trail_start_atr * atr_current:
                self.position["best_price"] = best_price
                return
            proposed_sl = best_price + trail_atr_mult * atr_current
            current_sl = self.position.get("sl")
            new_sl = proposed_sl if current_sl is None else min(float(current_sl), proposed_sl)
            improve = current_sl is None or (float(current_sl) - new_sl >= TRAIL_UPDATE_MIN_STEP_ATR * atr_current)
        self.position["best_price"] = best_price
        if not improve:
            return
        self._modify_position_sltp(new_sl)

    def _mc_dropout_uncertainty(
        self,
        short: torch.Tensor,
        mid: torch.Tensor,
        long: torch.Tensor,
        slow: torch.Tensor,
        trend_short: torch.Tensor,
        trend_long: torch.Tensor,
        close_short: torch.Tensor,
        close_long: torch.Tensor,
    ) -> tuple[float, float]:
        if not ENABLE_MC_DROPOUT_UNCERTAINTY or MC_DROPOUT_PASSES <= 0:
            return 0.0, 0.0
        self.model.train()
        trade_samples: list[torch.Tensor] = []
        up_samples: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(int(MC_DROPOUT_PASSES)):
                (
                    trade_logit,
                    dir_logit,
                    bounce_logit,
                    ttp_logit,
                    edge_pred,
                    qdist_pred,
                    p_trend_gate,
                    ksl_pred,
                    ktp_pred,
                    dist_pred,
                    break_pred,
                    p_trend_up,
                    transition_logit,
                ) = self.model(short, mid, long, slow, trend_short, trend_long, close_short, close_long)
                trade_samples.append(torch.sigmoid(trade_logit).squeeze())
                up_samples.append(torch.sigmoid(dir_logit).squeeze())
        self.model.eval()
        trade_stack = torch.stack(trade_samples)
        up_stack = torch.stack(up_samples)
        trade_std = float(trade_stack.std(unbiased=False).item())
        up_std = float(up_stack.std(unbiased=False).item())
        return trade_std, up_std

    def _close_position(self, price: float) -> None:
        if self.position is None:
            return
        direction = self.position["direction"]
        lot = self._normalize_volume(self.cfg.symbol, float(self.position.get("volume", self.cfg.lots)))
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.cfg.symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_SELL if direction > 0 else mt5.ORDER_TYPE_BUY,
            "position": self.position["ticket"],
            "price": price,
            "deviation": self.cfg.slippage,
            "comment": self.cfg.comment,
            "magic": self.cfg.magic,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error("Failed to close trade: %s", result)
            return
        logging.info("Closed %s at %.2f", "long" if direction > 0 else "short", price)
        self.position = None

    def _log_live(
        self,
        bar_ts: pd.Timestamp,
        close: float,
        spread: float,
        p_trade: float,
        p_entry_ttp_quick: float,
        p_trade_eff: float,
        edge_hat_atr: float,
        q10_hat: float,
        q50_hat: float,
        q90_hat: float,
        kelly_f: float,
        kelly_lot: float,
        eff_thr: float,
        p_trade_std: float,
        p_up: float,
        p_up_std: float,
        p_bounce: float,
        p_trend: float,
        p_transition: float,
        transition_dir: int,
        dist_ev_atr: float,
        dist_q10_atr: float,
        dist_q50_atr: float,
        dist_q75_atr: float,
        dist_q95_atr: float,
        dist_exit_cond: int,
        p_break: float,
        excess_up_atr: float,
        excess_dn_atr: float,
        atr: float,
        k_sl_raw: float,
        k_tp_raw: float,
        k_sl_used: float,
        k_tp_used: float,
        k_sl_before_extra: float,
        sl_extra_dollars: float,
        sl_extra_atr: float,
        k_sl_final: float,
        trail_start_atr: float,
        trail_atr_mult: float,
        sl: Optional[float],
        tp: Optional[float],
        lot: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        action: str,
        entry_side: int,
        exit_now: bool,
        exit_streak: int,
        exit_reason: str,
        hold_bars: int,
    ) -> None:
        with self.log_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    bar_ts.isoformat(),
                    f"{close:.5f}",
                    f"{spread:.5f}",
                    f"{p_trade:.6f}",
                    f"{p_entry_ttp_quick:.6f}",
                    f"{p_trade_eff:.6f}",
                    f"{edge_hat_atr:.6f}",
                    f"{q10_hat:.6f}",
                    f"{q50_hat:.6f}",
                    f"{q90_hat:.6f}",
                    f"{kelly_f:.6f}",
                    f"{kelly_lot:.4f}",
                    f"{eff_thr:.6f}",
                    f"{p_trade_std:.6f}",
                    f"{p_up:.6f}",
                    f"{p_up_std:.6f}",
                    f"{p_bounce:.6f}",
                    f"{p_trend:.6f}",
                    f"{p_transition:.6f}",
                    transition_dir,
                    f"{dist_ev_atr:.6f}",
                    f"{dist_q10_atr:.6f}",
                    f"{dist_q50_atr:.6f}",
                    f"{dist_q75_atr:.6f}",
                    f"{dist_q95_atr:.6f}",
                    int(dist_exit_cond),
                    f"{p_break:.6f}",
                    f"{excess_up_atr:.6f}",
                    f"{excess_dn_atr:.6f}",
                    f"{atr:.6f}",
                    f"{k_sl_raw:.4f}",
                    f"{k_tp_raw:.4f}",
                    f"{k_sl_used:.4f}",
                    f"{k_tp_used:.4f}",
                    f"{k_sl_before_extra:.4f}",
                    f"{sl_extra_dollars:.2f}",
                    f"{sl_extra_atr:.4f}",
                    f"{k_sl_final:.4f}",
                    f"{trail_start_atr:.4f}",
                    f"{trail_atr_mult:.4f}",
                    "" if sl is None else f"{sl:.5f}",
                    "" if tp is None else f"{tp:.5f}",
                    f"{lot:.2f}",
                    "" if sl_price is None else f"{sl_price:.5f}",
                    "" if tp_price is None else f"{tp_price:.5f}",
                    action,
                    entry_side,
                    int(exit_now),
                    exit_streak,
                    exit_reason,
                    hold_bars,
                ]
                )

    def _sync_position_with_mt5(self) -> None:
        if mt5 is None or self.dry_run or self.position is None:
            return
        try:
            pos = self._get_mt5_position()
            if pos is None:
                logging.info("No MT5 position for magic=%s; clearing local position.", self.cfg.magic)
                self.position = None
                return
            self.position["ticket"] = int(getattr(pos, "ticket", 0))
            self.position["direction"] = 1 if self._pos_side(pos) == "BUY" else -1
            self.position["entry_price"] = float(getattr(pos, "price_open", self.position.get("entry_price", 0.0)))
        except Exception as exc:
            logging.warning("MT5 position sync failed: %s", exc)

    def _update_eff_thresholds(self, p_trade_eff: float, trend_mode: bool) -> None:
        if not USE_ADAPTIVE_EFF_THR:
            return
        buf = self.eff_buf_trend if trend_mode else self.eff_buf_mr
        buf.append(float(p_trade_eff))
        if self.bar_index < int(EFF_THR_WARMUP_BARS):
            return
        if (self.bar_index - self.last_thr_update_bar) < int(EFF_THR_UPDATE_EVERY):
            return
        self.last_thr_update_bar = self.bar_index
        if EFF_THR_MODE == "rolling_meanstd":
            arr = np.asarray(buf, dtype=np.float64)
            if arr.size:
                thr = float(arr.mean() + arr.std())
            else:
                thr = float(EFF_THR_STATIC_TREND if trend_mode else EFF_THR_STATIC_MR)
        else:
            thr = _safe_percentile(list(buf), 100.0 * (EFF_THR_PERCENTILE_TREND if trend_mode else EFF_THR_PERCENTILE_MR))
        thr = _clamp(thr, EFF_THR_MIN, EFF_THR_MAX)
        if trend_mode:
            self.eff_thr_trend = thr
        else:
            self.eff_thr_mr = thr
        if EFF_THR_DEBUG_EVERY > 0 and (self.bar_index % int(EFF_THR_DEBUG_EVERY) == 0):
            p50 = _safe_percentile(list(buf), 50.0)
            p75 = _safe_percentile(list(buf), 75.0)
            p90 = _safe_percentile(list(buf), 90.0)
            p95 = _safe_percentile(list(buf), 95.0)
            logging.info(
                "EffThr update: mode=%s n=%s p50=%.3f p75=%.3f p90=%.3f p95=%.3f thr=%.3f",
                "TREND" if trend_mode else "MR",
                len(buf),
                p50,
                p75,
                p90,
                p95,
                thr,
            )

    def run(self) -> None:
        self._ensure_connection()
        lookback = max(
            LONG_LEN,
            MID_LEN,
            SHORT_LEN,
            max(CORR_WINDOWS) if CORR_WINDOWS else 1,
            max(RET_WINDOWS) if RET_WINDOWS else 1,
            VOL_WINDOW,
            CHANNEL_WINDOW_MINUTES,
            LEVEL_LOOKBACK_DAYS * 1440,
        )
        lookback = int(max(lookback, 200))
        while True:
            try:
                loop_start = time.time()
                fetch_start = time.perf_counter()
                xau = _fetch_m1(self.cfg.symbol, lookback)
                xag = _fetch_m1(XAG_SYMBOL, lookback)
                dxy = _fetch_m1(self.usd_symbol, lookback)
                fetch_ms = (time.perf_counter() - fetch_start) * 1000.0
                if xau.empty:
                    time.sleep(LIVE_POLL_SECONDS)
                    continue
                if len(xau) < 2:
                    time.sleep(LIVE_POLL_SECONDS)
                    continue
                closed_bar_ts = xau["bar_dt"].iloc[-2]
                forming_bar_ts = xau["bar_dt"].iloc[-1]
                prev_seen = self.last_seen_bar_ts
                if self.last_seen_bar_ts is not None and closed_bar_ts <= self.last_seen_bar_ts:
                    new_closed_bar = False
                else:
                    new_closed_bar = True
                    self.last_seen_bar_ts = closed_bar_ts
                if DEBUG_TREND_GATE_LIVE and prev_seen is not None and closed_bar_ts <= prev_seen:
                    logging.warning(
                        "TREND_GATE_LIVE timestamps not advancing: closed=%s last_seen=%s",
                        closed_bar_ts,
                        prev_seen,
                    )
                if new_closed_bar:
                    self.bar_index += 1
                if new_closed_bar:
                    self._sync_position_with_mt5()
                if ENABLE_LIVE_FEATURE_TIMING and new_closed_bar:
                    global _TIMING_ACTIVE
                    _TIMING_ACTIVE = True
                    _reset_timing()
                    _TIMING_MS["fetch_data"] = fetch_ms
                    _TIMING_STARTS["total"] = time.perf_counter() - (fetch_ms / 1000.0)
                elif ENABLE_LIVE_FEATURE_TIMING:
                    _TIMING_ACTIVE = False

                tick = mt5.symbol_info_tick(self.cfg.symbol)
                tick_price = None
                if tick is not None:
                    tick_price = float(tick.last) if tick.last > 0 else float((tick.bid + tick.ask) / 2.0)
                    logging.info(
                        "LIVE tick symbol=%s time_msc=%s bid=%.5f ask=%.5f last=%.5f",
                        self.cfg.symbol,
                        tick.time_msc,
                        tick.bid,
                        tick.ask,
                        tick.last,
                    )
                live_spread = self._get_live_spread()
                if live_spread > 0.0 and SPREAD_POINTS_TO_PRICE > 0.0:
                    xau["spread"] = pd.to_numeric(xau["spread"], errors="coerce").astype(float)
                    xau.loc[xau.index[-1], "spread"] = live_spread / SPREAD_POINTS_TO_PRICE
                if tick_price is not None and tick_price > 0:
                    xau.loc[xau.index[-1], "close"] = tick_price

                if self.cached_df is None or new_closed_bar:
                    if new_closed_bar:
                        start_timer("feature_frame")
                    ff_start = time.perf_counter()
                    logging.info(
                        "Feature frame: start (xau=%s xag=%s dxy=%s)",
                        len(xau),
                        len(xag),
                        len(dxy),
                    )
                    df, feature_cols = _build_feature_frame_from_sources(
                        xau,
                        xag,
                        dxy,
                        log_progress=LIVE_FEATURE_PROGRESS,
                        live_fill=True,
                        progress_every=LIVE_FEATURE_PROGRESS_EVERY,
                    )
                    ff_ms = (time.perf_counter() - ff_start) * 1000.0
                    logging.info("Feature frame: done in %.1fms rows=%s cols=%s", ff_ms, len(df), len(feature_cols))
                    if new_closed_bar:
                        stop_timer("feature_frame")
                    self.cached_df = df
                    self.cached_feature_cols = feature_cols
                    self.cached_full_ts = closed_bar_ts
                else:
                    if self.cached_df is None or self.cached_feature_cols is None:
                        if new_closed_bar:
                            start_timer("feature_frame")
                        ff_start = time.perf_counter()
                        logging.info(
                            "Feature frame: start (xau=%s xag=%s dxy=%s)",
                            len(xau),
                            len(xag),
                            len(dxy),
                        )
                        df, feature_cols = _build_feature_frame_from_sources(
                            xau,
                            xag,
                            dxy,
                            log_progress=LIVE_FEATURE_PROGRESS,
                            live_fill=True,
                            progress_every=LIVE_FEATURE_PROGRESS_EVERY,
                        )
                        ff_ms = (time.perf_counter() - ff_start) * 1000.0
                        logging.info("Feature frame: done in %.1fms rows=%s cols=%s", ff_ms, len(df), len(feature_cols))
                        if new_closed_bar:
                            stop_timer("feature_frame")
                        self.cached_df = df
                        self.cached_feature_cols = feature_cols
                        self.cached_full_ts = closed_bar_ts
                    else:
                        if new_closed_bar:
                            start_timer("fast_features")
                        df = _fast_update_live_features(self.cached_df, xau, xag, dxy)
                        if new_closed_bar:
                            stop_timer("fast_features")
                        feature_cols = self.cached_feature_cols
                if new_closed_bar:
                    start_timer("sequence_build")
                (
                    short_seq,
                    mid_seq,
                    long_seq,
                    slow_seq,
                    trend_short_seq,
                    trend_long_seq,
                    close_short_seq,
                    close_long_seq,
                    idx,
                ) = _build_live_sequences(df, feature_cols, self.norm_params)
                if new_closed_bar:
                    stop_timer("sequence_build")

                short = torch.from_numpy(short_seq).unsqueeze(0).to(DEVICE)
                mid = torch.from_numpy(mid_seq).unsqueeze(0).to(DEVICE)
                long = torch.from_numpy(long_seq).unsqueeze(0).to(DEVICE)
                slow = torch.from_numpy(slow_seq).unsqueeze(0).to(DEVICE)
                trend_short = torch.from_numpy(trend_short_seq).unsqueeze(0).to(DEVICE)
                trend_long = torch.from_numpy(trend_long_seq).unsqueeze(0).to(DEVICE)
                close_short = torch.from_numpy(close_short_seq).unsqueeze(0).to(DEVICE)
                close_long = torch.from_numpy(close_long_seq).unsqueeze(0).to(DEVICE)

                if new_closed_bar:
                    start_timer("model_inference")
                with torch.no_grad():
                    (
                        trade_logit,
                        dir_logit,
                        bounce_logit,
                        ttp_logit,
                        edge_pred,
                        qdist_pred,
                        p_trend_gate,
                        ksl_pred,
                        ktp_pred,
                        dist_pred,
                        break_pred,
                        p_trend_up,
                        transition_logit,
                    ) = self.model(short, mid, long, slow, trend_short, trend_long, close_short, close_long)
                    p_trade = float(torch.sigmoid(trade_logit).item())
                    p_up = float(torch.sigmoid(dir_logit).item())
                    p_bounce = float(torch.sigmoid(bounce_logit).item())
                    p_trend_gate_val = float(p_trend_gate.item())
                    p_trend_up_val = float(p_trend_up.item())
                    p_entry_ttp_quick = float(torch.sigmoid(ttp_logit).item())
                    p_transition = float(torch.sigmoid(transition_logit).item())
                    k_sl_raw = float(ksl_pred.item())
                    k_tp_raw = float(ktp_pred.item())
                    p_break = float(break_pred.item())
                if new_closed_bar:
                    stop_timer("model_inference")

                if not USE_TTP_HEAD:
                    p_entry_ttp_quick = 1.0
                if not USE_TRANSITION_HEAD:
                    p_transition = 0.0
                edge_hat_atr = float(edge_pred.item()) if USE_EDGE_HEAD else 0.0
                if USE_QDIST_HEAD:
                    qdist_raw = qdist_pred.squeeze(0).detach().cpu().numpy()
                    q10_hat = float(qdist_raw[_qdist_index(0.1)])
                    q50_hat = float(qdist_raw[_qdist_index(0.5)])
                    q90_hat = float(qdist_raw[_qdist_index(0.9)])
                else:
                    q10_hat = 0.0
                    q50_hat = 0.0
                    q90_hat = 0.0
                kelly_f = 0.0
                kelly_lot = 0.0

                if USE_LEARNED_ENTRY_COMBINER and hasattr(self.model, "entry_combiner"):
                    trade_feat = trade_logit if COMBINER_USE_LOGITS else torch.sigmoid(trade_logit)
                    trend_feat = _logit_from_prob(p_trend_gate) if COMBINER_USE_LOGITS else p_trend_gate
                    trend_dir_feat = _logit_from_prob(p_trend_up) if COMBINER_USE_LOGITS else p_trend_up
                    dir_feat = dir_logit if COMBINER_USE_LOGITS else torch.sigmoid(dir_logit)
                    feats = [trade_feat, trend_feat, trend_dir_feat, dir_feat]
                    if COMBINER_INCLUDE_TTP:
                        ttp_feat = ttp_logit if COMBINER_USE_LOGITS else torch.sigmoid(ttp_logit)
                        ttp_feat = torch.clamp(ttp_feat, -COMBINER_TTP_CLIP, COMBINER_TTP_CLIP)
                        feats.append(ttp_feat)
                    if COMBINER_INCLUDE_EDGE:
                        edge_feat = edge_pred
                        edge_feat = torch.clamp(edge_feat, -COMBINER_EDGE_CLIP, COMBINER_EDGE_CLIP)
                        feats.append(edge_feat)
                    if COMBINER_INCLUDE_QDIST:
                        qdist_feat = torch.clamp(qdist_pred, -COMBINER_QDIST_CLIP, COMBINER_QDIST_CLIP)
                        feats.extend([qdist_feat[:, i] for i in range(qdist_feat.shape[1])])
                    if COMBINER_INCLUDE_TAIL:
                        tail_asym = qdist_pred[:, -1] - qdist_pred[:, 0]
                        tail_asym = torch.clamp(tail_asym, -COMBINER_TAIL_CLIP, COMBINER_TAIL_CLIP)
                        feats.append(tail_asym)
                    if COMBINER_INCLUDE_CONTEXT:
                        vol_val = float(df.loc[idx, "vol_30m"]) if "vol_30m" in df.columns else 0.0
                        spread_val = float(df.loc[idx, "xau_spread"]) if "xau_spread" in df.columns else 0.0
                        strength_val = 0.0
                        feats.extend(
                            [
                                torch.tensor([vol_val], device=DEVICE),
                                torch.tensor([spread_val], device=DEVICE),
                                torch.tensor([strength_val], device=DEVICE),
                            ]
                        )
                    comb_in = torch.stack(feats, dim=-1).unsqueeze(0)
                    combiner_logit = self.model.entry_combiner(comb_in).squeeze()
                    p_trade_eff = float(torch.sigmoid(combiner_logit).item())
                else:
                    p_trade_eff = p_trade

                if False:
                    logging.info("TTP gate: p_trade=%.4f p_ttp=%.4f p_trade_eff=%.4f", p_trade, p_entry_ttp_quick, p_trade_eff)

                p_trend = p_trend_gate_val
                trend_mode = self.trend_mode
                trend_weight = 1.0 if trend_mode else 0.0
                if new_closed_bar:
                    if DEBUG_TREND_GATE_LIVE and (self.bar_index % int(DEBUG_TREND_GATE_EVERY_N_BARS) == 0):
                        close_tail = close_short_seq[-5:] if close_short_seq.size >= 5 else close_short_seq
                        logging.info(
                            "TREND_GATE_LIVE bar=%s close=%.5f p_trend=%.4f p_trend_up=%.4f close_tail=%s",
                            closed_bar_ts,
                            float(close_short_seq[-1]),
                            p_trend,
                            p_trend_up_val,
                            np.array2string(close_tail, precision=5, separator=","),
                        )
                        if close_short_seq.size >= 2 and np.isclose(close_short_seq[-1], close_short_seq[-2]):
                            logging.warning("TREND_GATE_LIVE close unchanged at %.5f", float(close_short_seq[-1]))
                    trend_mode = self._update_regime_state(p_trend_gate_val)
                    if REGIME_BLEND_MODE:
                        trend_weight = _clamp(p_trend_gate_val, 0.0, 1.0)
                    else:
                        trend_weight = 1.0 if trend_mode else 0.0
                    if TREND_DIAG_EVERY_BARS > 0 and (self.bar_index - self.last_trend_log_bar) >= TREND_DIAG_EVERY_BARS:
                        diag = _trend_diag_from_seq(close_short_seq, close_long_seq, trend_short_seq=trend_short_seq)
                        if diag:
                            logging.info(
                                "Trend diag: mode=%s p_trend=%.3f slope=%.4f push_up=%.3f push_dn=%.3f pull_dn=%.3f pull_up=%.3f ratio_up=%.3f ratio_dn=%.3f switches=%s",
                                "TREND" if trend_mode else "MR",
                                p_trend,
                                diag.get("slope_norm", 0.0),
                                diag.get("push_up_atr", 0.0),
                                diag.get("push_dn_atr", 0.0),
                                diag.get("pullback_dn_atr", 0.0),
                                diag.get("pullback_up_atr", 0.0),
                                diag.get("ratio_up", 0.0),
                                diag.get("ratio_dn", 0.0),
                                self.trend_switches,
                            )
                        self.last_trend_log_bar = self.bar_index
                    self._update_eff_thresholds(p_trade_eff, trend_mode)

                transition_dir = 0
                transition_active = False
                slopes_norm = None
                if new_closed_bar:
                    transition_dir, transition_active, slopes_norm = _transition_dir_from_seq(
                        close_short_seq,
                        trend_short_seq=trend_short_seq,
                    )
                    if (
                        USE_TRANSITION_HEAD
                        and TRANSITION_DEBUG_EVERY_BARS > 0
                        and (self.bar_index - self.last_transition_log_bar) >= TRANSITION_DEBUG_EVERY_BARS
                    ):
                        slope_short = float(slopes_norm[0]) if slopes_norm is not None and slopes_norm.size else 0.0
                        slope_long = float(slopes_norm[-1]) if slopes_norm is not None and slopes_norm.size else 0.0
                        logging.info(
                            "Transition diag: p_transition=%.3f dir=%s slope_short=%.4f slope_long=%.4f active=%s",
                            p_transition,
                            transition_dir,
                            slope_short,
                            slope_long,
                            transition_active,
                        )
                        self.last_transition_log_bar = self.bar_index

                p_trade_std = self.last_p_trade_std
                p_up_std = self.last_p_up_std
                if new_closed_bar and ENABLE_MC_DROPOUT_UNCERTAINTY:
                    p_trade_std, p_up_std = self._mc_dropout_uncertainty(
                        short, mid, long, slow, trend_short, trend_long, close_short, close_long
                    )
                    self.last_p_trade_std = p_trade_std
                    self.last_p_up_std = p_up_std

                direction_for_scoring = 1 if p_up >= 0.5 else -1
                if USE_KELLY_SIZING and USE_QDIST_HEAD:
                    q10_a = q10_hat if direction_for_scoring > 0 else -q90_hat
                    q50_a = q50_hat if direction_for_scoring > 0 else -q50_hat
                    q90_a = q90_hat if direction_for_scoring > 0 else -q10_hat
                    kelly_lot, kelly_f = _compute_kelly_lot(p_trade_eff, q10_a, q50_a, q90_a)
                if DYNAMIC_ATR_MULTS:
                    k_sl_rule, k_tp_rule, trail_start_atr, trail_atr_mult = _dynamic_atr_mults(
                        p_trade=float(p_trade),
                        p_up=float(p_up),
                        p_trade_std=float(self.last_p_trade_std),
                        p_up_std=float(self.last_p_up_std),
                        direction=direction_for_scoring,
                    )
                else:
                    k_sl_rule = float(SL_ATR_MULT)
                    k_tp_rule = float(TP_ATR_MULT)
                    trail_start_atr = float(TRAIL_START_ATR)
                    trail_atr_mult = float(TRAIL_ATR_MULT)

                if LEARNED_SLTP:
                    k_sl_used = _clip_if(k_sl_raw, K_SL_MIN, K_SL_MAX, APPLY_K_CLAMPS)
                    k_tp_used = _clip_if(k_tp_raw, K_TP_MIN, K_TP_MAX, APPLY_K_CLAMPS)
                else:
                    k_sl_used = float(k_sl_rule)
                    k_tp_used = float(k_tp_rule)

                self.last_k_sl = k_sl_rule
                self.last_k_tp = k_tp_rule
                self.last_k_sl_raw = k_sl_raw
                self.last_k_tp_raw = k_tp_raw
                self.last_k_sl_used = k_sl_used
                self.last_k_tp_used = k_tp_used
                self.last_trail_start_atr = trail_start_atr
                self.last_trail_atr_mult = trail_atr_mult

                close = float(df.loc[idx, "xau_close"])
                excess_up_atr = float(df.loc[idx, "excess_up_atr"]) if "excess_up_atr" in df.columns else 0.0
                excess_dn_atr = float(df.loc[idx, "excess_dn_atr"]) if "excess_dn_atr" in df.columns else 0.0
                spread = float(df.loc[idx, "xau_spread"])
                action = "skip"
                hold_bars = 0
                entry_side = 0 if self.position is None else int(self.position.get("direction", 0))
                exit_now = False
                exit_reason = ""
                pnl_atr = 0.0
                dist_ev_atr = self.last_dist_ev_atr
                dist_q10_atr = self.last_dist_q10_atr
                dist_q50_atr = self.last_dist_q50_atr
                dist_q75_atr = self.last_dist_q75_atr
                dist_q95_atr = self.last_dist_q95_atr
                dist_exit_cond = self.last_dist_exit_cond
                eff_thr_trend = (
                    self.eff_thr_trend
                    if USE_ADAPTIVE_EFF_THR and self.bar_index >= int(EFF_THR_WARMUP_BARS) and len(self.eff_buf_trend)
                    else float(EFF_THR_STATIC_TREND)
                )
                eff_thr_mr = (
                    self.eff_thr_mr
                    if USE_ADAPTIVE_EFF_THR and self.bar_index >= int(EFF_THR_WARMUP_BARS) and len(self.eff_buf_mr)
                    else float(EFF_THR_STATIC_MR)
                )
                eff_thr_used = eff_thr_trend if trend_mode else eff_thr_mr

                if new_closed_bar:
                    atr_current = _atr(xau, ATR_PERIOD)
                    self.last_atr = atr_current
                    bid = float(tick.bid) if tick is not None and tick.bid else None
                    ask = float(tick.ask) if tick is not None and tick.ask else None
                    dist_pred_atr = dist_pred.squeeze(0).detach().cpu().numpy()
                    h_idx = _horizon_index(DIST_EXIT_EV_HORIZON)
                    q10_idx = _quantile_index(10)
                    q50_idx = _quantile_index(50)
                    q75_idx = _quantile_index(75)
                    q95_idx = _quantile_index(95)
                    dist_q10_atr = float(dist_pred_atr[h_idx, q10_idx])
                    dist_q50_atr = float(dist_pred_atr[h_idx, q50_idx])
                    dist_q75_atr = float(dist_pred_atr[h_idx, q75_idx])
                    dist_q95_atr = float(dist_pred_atr[h_idx, q95_idx])
                    w_low, w_med, w_high = DIST_EXIT_EV_WEIGHTS
                    dist_ev_atr = float(w_low * dist_q10_atr + w_med * dist_q50_atr + w_high * dist_q75_atr)
                    self.last_dist_ev_atr = dist_ev_atr
                    self.last_dist_q10_atr = dist_q10_atr
                    self.last_dist_q50_atr = dist_q50_atr
                    self.last_dist_q75_atr = dist_q75_atr
                    self.last_dist_q95_atr = dist_q95_atr
                    # === ER VETO (MR ENTRY GUARD) ===
                    er_veto = False
                    er_5m = 0.0
                    signed_er_5m = 0.0
                    timeframe_minutes = max(1, _timeframe_to_minutes(self.cfg.timeframe))
                    bars_per_window = max(2, int(round(ER_WINDOW_MINUTES / timeframe_minutes)))
                    window_len = bars_per_window + 1
                    closes_series = xau["close"].to_numpy(dtype=float)
                    if ER_BLOCK_REQUIRE_CLOSED_BAR and closes_series.size > 0:
                        closes_series = closes_series[:-1]
                    if window_len >= 3 and closes_series.size >= window_len:
                        closes_window = closes_series[-window_len:]
                        er_5m, signed_er_5m = _compute_er_and_signed_er(closes_window)
                        er_veto = abs(signed_er_5m) > ER_BLOCK_ABS_THR
                        if er_veto and ER_DEBUG_LOG:
                            logging.info(
                                "ER veto: signed_ER=%+.2f ER=%.2f N=%s -> MR entry blocked",
                                signed_er_5m,
                                er_5m,
                                window_len - 1,
                            )
                    k_sl_before_extra = float(k_sl_used)
                    sl_extra_atr = 0.0
                    k_sl_final = k_sl_before_extra
                    if SL_EXTRA_DOLLARS > 0.0 and atr_current > 0.0:
                        sl_extra_atr = SL_EXTRA_DOLLARS / (atr_current + 1e-8)
                        sl_extra_atr = min(sl_extra_atr, MAX_SL_EXTRA_ATR)
                        k_sl_final = k_sl_before_extra + sl_extra_atr
                    self.last_k_sl_before_extra = k_sl_before_extra
                    self.last_sl_extra_atr = sl_extra_atr
                    self.last_k_sl_final = k_sl_final
                    k_sl_trade = k_sl_final
                    k_tp_trade = k_tp_used
                    trail_start_eff = trail_start_atr
                    trail_atr_eff = trail_atr_mult
                    if trend_weight > 0.0:
                        k_sl_trade = k_sl_final * _lerp(1.0, TREND_KSL_MULT, trend_weight)
                        k_tp_trade = k_tp_used * _lerp(1.0, TREND_KTP_MULT, trend_weight)
                        trail_start_eff = _lerp(trail_start_atr, TREND_TRAIL_START_ATR, trend_weight)
                        trail_atr_eff = _lerp(trail_atr_mult, TREND_TRAIL_ATR_MULT, trend_weight)
                    self.last_k_sl_used = k_sl_trade
                    self.last_k_tp_used = k_tp_trade
                    self.last_trail_start_atr = trail_start_eff
                    self.last_trail_atr_mult = trail_atr_eff
                    if self.position is not None:
                        self._update_trailing_stop(
                            atr_current,
                            bid,
                            ask,
                            close,
                            trail_start_eff,
                            trail_atr_eff,
                        )
                    symbol_info = _get_symbol_info(self.cfg.symbol)
                    tick_now = _get_tick(self.cfg.symbol)
                    positions = _positions_by_magic(self.cfg.symbol, self.cfg.magic)
                    if not positions:
                        self.position = None
                    else:
                        if tick_now is None or symbol_info is None:
                            logging.warning("Skip SL/TP modify: tick or symbol_info unavailable.")
                        else:
                            bid_now = float(tick_now.bid) if tick_now.bid else 0.0
                            ask_now = float(tick_now.ask) if tick_now.ask else 0.0
                            min_dist = _min_stop_distance(symbol_info)
                            digits = int(getattr(symbol_info, "digits", 2) or 2)
                            sl_dist = float(k_sl_trade) * atr_current if ENABLE_SL else 0.0
                            tp_dist = float(k_tp_trade) * atr_current if ENABLE_TP else 0.0
                            for pos in positions:
                                pos_ticket = int(getattr(pos, "ticket", 0) or 0)
                                pos_type = int(getattr(pos, "type", 0) or 0)
                                pos_open = float(getattr(pos, "price_open", 0.0) or 0.0)
                                current_sl = float(getattr(pos, "sl", 0.0) or 0.0)
                                current_tp = float(getattr(pos, "tp", 0.0) or 0.0)
                                if pos_ticket == 0:
                                    continue
                                if pos_type == mt5.POSITION_TYPE_BUY:
                                    proposed_sl = pos_open - sl_dist if ENABLE_SL else 0.0
                                    proposed_tp = pos_open + tp_dist if ENABLE_TP else 0.0
                                else:
                                    proposed_sl = pos_open + sl_dist if ENABLE_SL else 0.0
                                    proposed_tp = pos_open - tp_dist if ENABLE_TP else 0.0
                                proposed_sl = _round_price(proposed_sl, digits) if proposed_sl else 0.0
                                proposed_tp = _round_price(proposed_tp, digits) if proposed_tp else 0.0
                                sl_ok, tp_ok, sl_reason, tp_reason = _validate_sltp_for_position(
                                    pos_type, bid_now, ask_now, proposed_sl, proposed_tp, min_dist
                                )
                                if proposed_sl and not sl_ok:
                                    logging.warning("Disabled SL: %s", sl_reason)
                                    proposed_sl = 0.0
                                if proposed_tp and not tp_ok:
                                    logging.warning("Disabled TP: %s", tp_reason)
                                    proposed_tp = 0.0
                                point = float(getattr(symbol_info, "point", 0.0) or 0.0)
                                change_sl = (
                                    proposed_sl != 0.0
                                    and abs(proposed_sl - (current_sl or 0.0)) >= (point * 2)
                                ) or (proposed_sl == 0.0 and (current_sl or 0.0) != 0.0)
                                change_tp = (
                                    proposed_tp != 0.0
                                    and abs(proposed_tp - (current_tp or 0.0)) >= (point * 2)
                                ) or (proposed_tp == 0.0 and (current_tp or 0.0) != 0.0)
                                if not (change_sl or change_tp):
                                    continue
                                side = "BUY" if pos_type == mt5.POSITION_TYPE_BUY else "SELL"
                                logging.info(
                                    "SLTP modify bar: ticket=%s side=%s bid=%.5f ask=%.5f open=%.5f sl=%.5f tp=%.5f "
                                    "proposed_sl=%.5f proposed_tp=%.5f min_dist=%.5f digits=%s",
                                    pos_ticket,
                                    side,
                                    bid_now,
                                    ask_now,
                                    pos_open,
                                    current_sl,
                                    current_tp,
                                    proposed_sl,
                                    proposed_tp,
                                    min_dist,
                                    digits,
                                )
                                req, res = _maybe_send_sltp_modify(
                                    pos_ticket,
                                    self.cfg.symbol,
                                    proposed_sl,
                                    proposed_tp,
                                    self.cfg.magic,
                                    "SLTP update",
                                )
                                if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                                    logging.error(
                                        "SLTP modify failed: retcode=%s comment=%s last_error=%s req=%s",
                                        getattr(res, "retcode", None) if res is not None else None,
                                        getattr(res, "comment", None) if res is not None else None,
                                        mt5.last_error() if mt5 is not None else None,
                                        req,
                                    )
                    if self.position is None:
                        use_trend_entry = trend_mode if not REGIME_BLEND_MODE else (p_trend_gate_val >= 0.5)
                        if use_trend_entry:
                            use_transition = (
                                USE_TRANSITION_HEAD
                                and TRANSITION_USE_IN_TREND_MODE
                                and p_transition >= TRANSITION_MIN_PROB
                                and transition_dir != 0
                                and ((not TRANSITION_REQUIRE_TREND) or (p_trend >= TRANSITION_TREND_MIN))
                            )
                            direction = transition_dir if use_transition else (1 if p_trend_up_val >= 0.5 else -1)
                            if use_transition:
                                logging.info(
                                    "Transition entry: p_transition=%.3f dir=%s p_trend=%.3f",
                                    p_transition,
                                    direction,
                                    p_trend,
                                )
                            if TREND_REQUIRE_P_TRADE and p_trade_eff < eff_thr_trend:
                                pass
                            else:
                                if TTP_USE_AS_GATE and TTP_GATE_MODE == "hard" and p_entry_ttp_quick < TTP_MIN_PROB:
                                    pass
                                else:
                                    tick = mt5.symbol_info_tick(self.cfg.symbol)
                                    price = tick.ask if direction > 0 else tick.bid
                                    if USE_KELLY_SIZING and USE_QDIST_HEAD:
                                        q10_a = q10_hat if direction > 0 else -q90_hat
                                        q50_a = q50_hat if direction > 0 else -q50_hat
                                        q90_a = q90_hat if direction > 0 else -q10_hat
                                        lot_override, kelly_f = _compute_kelly_lot(p_trade_eff, q10_a, q50_a, q90_a)
                                    else:
                                        lot_override = _compute_lot(p_trade_eff, p_trend_up_val, direction)
                                    if self._open_position(
                                        direction,
                                        price,
                                        atr_current,
                                        lot_override=lot_override,
                                        sl_mult=k_sl_trade,
                                        tp_mult=k_tp_trade,
                                    ):
                                        action = "open"
                                        entry_side = direction
                        else:
                            long_ok = p_up >= 0.5 and p_up >= LIVE_P_UP_MIN
                            short_ok = p_up < 0.5 and p_up <= LIVE_P_UP_SHORT_MAX
                            if p_trade_eff >= eff_thr_mr and (long_ok or short_ok):
                                direction = 1 if p_up >= 0.5 else -1
                                if TTP_USE_AS_GATE and TTP_GATE_MODE == "hard" and p_entry_ttp_quick < TTP_MIN_PROB:
                                    continue
                                if er_veto:
                                    action = "er_block"
                                    entry_side = 0
                                    continue
                                tick = mt5.symbol_info_tick(self.cfg.symbol)
                                price = tick.ask if direction > 0 else tick.bid
                                if USE_KELLY_SIZING and USE_QDIST_HEAD:
                                    q10_a = q10_hat if direction > 0 else -q90_hat
                                    q50_a = q50_hat if direction > 0 else -q50_hat
                                    q90_a = q90_hat if direction > 0 else -q10_hat
                                    lot_override, kelly_f = _compute_kelly_lot(p_trade_eff, q10_a, q50_a, q90_a)
                                else:
                                    lot_override = _compute_lot(p_trade_eff, p_up, direction)
                                if self._open_position(
                                    direction,
                                    price,
                                    atr_current,
                                    lot_override=lot_override,
                                    sl_mult=k_sl_trade,
                                    tp_mult=k_tp_trade,
                                ):
                                    action = "open"
                                    entry_side = direction
                    else:
                        self.position["hold_bars"] = int(self.position.get("hold_bars", 0)) + 1
                        hold_bars = int(self.position["hold_bars"])
                        direction = int(self.position.get("direction", 0))
                        if hold_bars < 1:
                            exit_now = False
                            exit_reason = "min_hold"
                            self.position["exit_streak"] = 0
                        else:
                            entry_price = float(self.position.get("entry_price", close))
                            mark_price = None
                            if direction > 0:
                                mark_price = bid if bid is not None and bid > 0 else close
                            else:
                                mark_price = ask if ask is not None and ask > 0 else close
                            pnl_ok = True
                            if DIST_EXIT_ONLY_IF_POSITIVE_PNL:
                                if atr_current <= 0.0 or mark_price is None:
                                    pnl_ok = False
                                else:
                                    pnl_atr = (mark_price - entry_price) / (atr_current + 1e-8)
                                    if direction < 0:
                                        pnl_atr = (entry_price - mark_price) / (atr_current + 1e-8)
                                    pnl_ok = pnl_atr > 0.0

                            transition_exit_now = False
                            if USE_TRANSITION_HEAD and TRANSITION_EXIT_ENABLE:
                                if (
                                    ((not TRANSITION_EXIT_REQUIRE_IN_TREND) or (p_trend >= TRANSITION_EXIT_TREND_MIN))
                                    and p_transition >= TRANSITION_EXIT_MIN_PROB
                                    and transition_dir != 0
                                    and direction != transition_dir
                                    and hold_bars >= TRANSITION_EXIT_MIN_BARS_IN_TRADE
                                ):
                                    self.transition_exit_streak += 1
                                else:
                                    self.transition_exit_streak = 0
                                if self.transition_exit_streak >= TRANSITION_EXIT_CONFIRM_BARS:
                                    if TRANSITION_EXIT_ACTION == "close":
                                        exit_now = True
                                        exit_reason = "transition_flip"
                                        transition_exit_now = True
                                        logging.info(
                                            "Transition exit: side=%s p_transition=%.3f new_dir=%s",
                                            direction,
                                            p_transition,
                                            transition_dir,
                                        )
                                    else:
                                        if atr_current > 0.0 and ENABLE_SL:
                                            tighten_dist = float(TRANSITION_EXIT_TIGHTEN_ATR) * atr_current
                                            new_sl = entry_price - tighten_dist if direction > 0 else entry_price + tighten_dist
                                            self._modify_position_sltp(new_sl)
                                            exit_reason = "transition_tighten"
                                            logging.info(
                                                "Transition tighten: side=%s p_transition=%.3f new_dir=%s sl=%.5f",
                                                direction,
                                                p_transition,
                                                transition_dir,
                                                new_sl,
                                            )
                                        elif not ENABLE_SL:
                                            exit_reason = "transition_tighten_skipped"
                                        self.transition_exit_streak = 0

                            if not transition_exit_now:
                                dist_aligned = dist_pred_atr if direction > 0 else -dist_pred_atr
                                dist_q10_atr = float(dist_aligned[h_idx, q10_idx])
                                dist_q50_atr = float(dist_aligned[h_idx, q50_idx])
                                dist_q75_atr = float(dist_aligned[h_idx, q75_idx])
                                dist_q95_atr = float(dist_aligned[h_idx, q95_idx])
                                w_low, w_med, w_high = DIST_EXIT_EV_WEIGHTS
                                dist_ev_atr = float(w_low * dist_q10_atr + w_med * dist_q50_atr + w_high * dist_q75_atr)
                                self.last_dist_ev_atr = dist_ev_atr
                                self.last_dist_q10_atr = dist_q10_atr
                                self.last_dist_q50_atr = dist_q50_atr
                                self.last_dist_q75_atr = dist_q75_atr
                                self.last_dist_q95_atr = dist_q95_atr

                                prev_q_med = self.position.get("prev_q_med")
                                flat_cond = False
                                if prev_q_med is not None:
                                    flat_cond = dist_q50_atr <= float(prev_q_med) + DIST_EXIT_MEDIAN_FLAT_EPS
                                self.position["prev_q_med"] = dist_q50_atr

                                exit_signal = False
                                self.last_dist_exit_cond = int(dist_exit_cond)
                                if hold_bars >= DIST_EXIT_MIN_BARS_IN_TRADE:
                                    if DIST_EXIT_USE_EV and dist_ev_atr < DIST_EXIT_EV_THRESHOLD:
                                        exit_signal = True
                                    if dist_q10_atr < DIST_EXIT_DOWNSIDE_LIMIT:
                                        exit_signal = True
                                    if flat_cond:
                                        exit_signal = True

                                dist_exit_cond = int(exit_signal)
                                exit_signal = exit_signal and pnl_ok
                                if exit_signal:
                                    self.position["exit_streak"] = int(self.position.get("exit_streak", 0)) + 1
                                    exit_reason = "dist_exit_signal"
                                    exit_now = self.position["exit_streak"] >= self.exit_confirm_bars
                                else:
                                    self.position["exit_streak"] = 0
                                    exit_reason = ""

                        if hold_bars >= 1 and hold_bars >= MAX_HOLD_MINUTES:
                            exit_now = True
                            exit_reason = f"{exit_reason}|time_stop" if exit_reason else "time_stop"

                        if exit_now:
                            tick = mt5.symbol_info_tick(self.cfg.symbol)
                            price = tick.bid if direction > 0 else tick.ask
                            if self.dry_run:
                                self.position = None
                            else:
                                self._close_position(price)
                            action = "close"
                        else:
                            action = "hold"

                hold_bars = 0 if self.position is None else int(self.position.get("hold_bars", 0))
                entry_side = 0 if self.position is None else int(self.position.get("direction", 0))
                exit_streak = 0 if self.position is None else int(self.position.get("exit_streak", 0))
                sl_val = None if self.position is None else self.position.get("sl")
                tp_val = None if self.position is None else self.position.get("tp")
                lot_val = 0.0 if self.position is None else float(self.position.get("volume", 0.0))
                sl_price = sl_val
                tp_price = tp_val
                if self.position is None:
                    self.transition_exit_streak = 0
                self._log_live(
                    forming_bar_ts,
                    close,
                    spread,
                    p_trade,
                    p_entry_ttp_quick,
                    p_trade_eff,
                    edge_hat_atr,
                    q10_hat,
                    q50_hat,
                    q90_hat,
                    kelly_f,
                    kelly_lot,
                    eff_thr_used,
                    p_trade_std,
                    p_up,
                    p_up_std,
                    p_bounce,
                    p_trend,
                    p_transition,
                    transition_dir,
                    dist_ev_atr,
                    dist_q10_atr,
                    dist_q50_atr,
                    dist_q75_atr,
                    dist_q95_atr,
                    dist_exit_cond,
                    p_break,
                    excess_up_atr,
                    excess_dn_atr,
                    self.last_atr,
                    self.last_k_sl_raw,
                    self.last_k_tp_raw,
                    self.last_k_sl_used,
                    self.last_k_tp_used,
                    self.last_k_sl_before_extra,
                    SL_EXTRA_DOLLARS,
                    self.last_sl_extra_atr,
                    self.last_k_sl_final,
                    self.last_trail_start_atr,
                    self.last_trail_atr_mult,
                    sl_val,
                    tp_val,
                    lot_val,
                    sl_price,
                    tp_price,
                    action,
                    entry_side,
                    exit_now,
                    exit_streak,
                    exit_reason,
                    hold_bars,
                )
                if ENABLE_LIVE_FEATURE_TIMING and new_closed_bar:
                    stop_timer("total")
                    _log_timing(self.timing_writer, self.timing_step)
                    self.timing_step += 1
                    _TIMING_ACTIVE = False
                logging.info(
                    "LIVE forming=%s closed=%s close=%.5f spread=%.5f p_trade=%.4f p_ttp=%.4f p_trade_eff=%.4f p_up=%.4f p_bounce=%.4f p_trend=%.4f p_transition=%.4f transition_dir=%s dist_ev_atr=%.4f dist_q10=%.4f dist_q50=%.4f dist_q75=%.4f dist_q95=%.4f action=%s entry_side=%s exit_now=%s exit_streak=%s reason=%s hold_bars=%s loop_s=%.2f",
                    forming_bar_ts,
                    closed_bar_ts,
                    close,
                    spread,
                    p_trade,
                    p_entry_ttp_quick,
                    p_trade_eff,
                    p_up,
                    p_bounce,
                    p_trend,
                    p_transition,
                    transition_dir,
                    dist_ev_atr,
                    dist_q10_atr,
                    dist_q50_atr,
                    dist_q75_atr,
                    dist_q95_atr,
                    action,
                    entry_side,
                    int(exit_now),
                    exit_streak,
                    exit_reason,
                    hold_bars,
                    time.time() - loop_start,
                )
            except Exception as exc:
                logging.exception("Live loop error: %s", exc)
            time.sleep(LIVE_POLL_SECONDS)


def _load_checkpoint(path: Path) -> tuple[TrendMRModel, list[str], dict[str, tuple[float, float]]]:
    logging.info("CKPT: about to torch.load %s (map_location=%s)", path, DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    logging.info("CKPT: torch.load finished (%s)", path)
    feature_cols = ckpt.get("feature_cols")
    norm_params = ckpt.get("norm_params", {})
    if not feature_cols:
        raise RuntimeError("Checkpoint missing feature_cols.")
    _resolve_trend_indices(feature_cols)
    model = TrendMRModel(in_features=len(feature_cols), slow_in_features=len(feature_cols)).to(DEVICE)
    logging.info("CKPT: model instantiated (in_features=%s)", len(feature_cols))
    try:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            logging.warning("Checkpoint missing keys: %s", missing)
        if unexpected:
            logging.warning("Checkpoint unexpected keys: %s", unexpected)
    except RuntimeError as exc:
        logging.error("Checkpoint incompatible (new heads). Retrain required.")
        raise
    logging.info("CKPT: state_dict loaded")
    model.eval()
    return model, feature_cols, norm_params


def run_live(dry_run: bool, ckpt_path: Optional[str], exit_confirm_bars: int, max_hold_minutes: Optional[int]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("RUN_LIVE: entered run_live (dry_run=%s, ckpt_path=%s)", dry_run, ckpt_path)
    logging.info("TREND-GATE FIX: using raw close windows separate from model features.")
    logging.info("RUN_LIVE: torch version=%s cuda_available=%s device=%s", torch.__version__, torch.cuda.is_available(), DEVICE)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for live mode but is not available.")
    _ensure_output_dirs()
    if max_hold_minutes is not None:
        global MAX_HOLD_MINUTES
        MAX_HOLD_MINUTES = int(max_hold_minutes)
    config = _apply_overrides()
    effective = _effective_config()
    logging.info("Effective Config Summary:\n%s", json.dumps(effective, indent=2, default=str))
    logging.info("Live MAX_HOLD_MINUTES=%s", MAX_HOLD_MINUTES)
    logging.info("Learned SL/TP: enabled=%s apply_clamps=%s", LEARNED_SLTP, APPLY_K_CLAMPS)
    logging.info("POS_WEIGHTS trade=%.2f dir=%.2f bounce=%.2f", POS_W_TRADE, POS_W_DIR, POS_W_BOUNCE)
    ckpt = Path(ckpt_path) if ckpt_path else _latest_checkpoint_path()
    logging.info("after cache -> about to load checkpoint %s", ckpt)
    model, feature_cols, norm_params = _load_checkpoint(ckpt)
    logging.info("Transition head enabled=%s", USE_TRANSITION_HEAD)
    if CALIBRATE_THRESHOLDS_ON_STARTUP:
        try:
            logging.info("[CALIBRATION] start (use_full_dataset=%s)", CALIBRATION_USE_FULL_DATASET)
            if CALIBRATION_USE_FULL_DATASET:
                cache_prev = CACHE_FEATURES
                if not CALIBRATION_USE_FEATURE_CACHE:
                    globals()["CACHE_FEATURES"] = False
                calib_df, calib_cols = _build_feature_frame()
                globals()["CACHE_FEATURES"] = cache_prev
            else:
                lookback = max(
                    CALIBRATION_BARS,
                    LONG_LEN,
                    MID_LEN,
                    SHORT_LEN,
                    TREND_SHORT_LOOKBACK_BARS,
                    TREND_LONG_LOOKBACK_BARS,
                    SLOW_LEN,
                    max(CORR_WINDOWS) if CORR_WINDOWS else 1,
                    max(RET_WINDOWS) if RET_WINDOWS else 1,
                    VOL_WINDOW,
                    CHANNEL_WINDOW_MINUTES,
                    LEVEL_LOOKBACK_DAYS * 1440,
                )
                lookback = int(max(lookback, 200))
                xau = _fetch_m1(TARGET_SYMBOL, lookback)
                xag = _fetch_m1(XAG_SYMBOL, lookback)
                dxy = _fetch_m1(_select_usd_index_symbol(), lookback)
                calib_df, calib_cols = _build_feature_frame_from_sources(
                    xau,
                    xag,
                    dxy,
                    log_progress=False,
                    live_fill=True,
                    progress_every=LIVE_FEATURE_PROGRESS_EVERY,
                )
            logging.info("[CALIBRATION] built feature frame rows=%s cols=%s", len(calib_df), len(calib_cols))
            if calib_cols != feature_cols:
                logging.warning("[CALIBRATION] feature_cols mismatch; using checkpoint columns.")
                calib_cols = feature_cols
            stats = _calibrate_entry_thresholds_on_startup(
                calib_df,
                calib_cols,
                norm_params,
                model,
                use_full_dataset=CALIBRATION_USE_FULL_DATASET,
            )
            logging.info("[CALIBRATION] thresholds computed: %s", stats)
            if stats:
                thr_trend = stats.get("trend_p75", float("nan"))
                thr_mr = stats.get("mr_p75", float("nan"))
                if stats.get("trend_count", 0) < CALIBRATION_MIN_SAMPLES_PER_REGIME:
                    logging.warning("[CALIBRATION] TREND samples too low; using default threshold.")
                    thr_trend = LIVE_TRADE_EFF_THR_TREND
                if stats.get("mr_count", 0) < CALIBRATION_MIN_SAMPLES_PER_REGIME:
                    logging.warning("[CALIBRATION] MR samples too low; using default threshold.")
                    thr_mr = LIVE_TRADE_EFF_THR_MR
                thr_trend = _clip_thr(float(thr_trend))
                thr_mr = _clip_thr(float(thr_mr))
                globals()["LIVE_TRADE_EFF_THR_TREND"] = thr_trend
                globals()["LIVE_TRADE_EFF_THR_MR"] = thr_mr

                logging.info("[CALIBRATION] bars=%s pct=%s", CALIBRATION_BARS, CALIBRATION_PERCENTILE)
                logging.info(
                    "[CALIBRATION] TREND n=%s p50=%.4f p75=%.4f p90=%.4f p95=%.4f p99=%.4f min=%.4f max=%.4f mean=%.4f",
                    stats.get("trend_count"),
                    stats.get("trend_p50", float("nan")),
                    stats.get("trend_p75", float("nan")),
                    stats.get("trend_p90", float("nan")),
                    stats.get("trend_p95", float("nan")),
                    stats.get("trend_p99", float("nan")),
                    stats.get("trend_min", float("nan")),
                    stats.get("trend_max", float("nan")),
                    stats.get("trend_mean", float("nan")),
                )
                logging.info(
                    "[CALIBRATION] MR n=%s p50=%.4f p75=%.4f p90=%.4f p95=%.4f p99=%.4f min=%.4f max=%.4f mean=%.4f",
                    stats.get("mr_count"),
                    stats.get("mr_p50", float("nan")),
                    stats.get("mr_p75", float("nan")),
                    stats.get("mr_p90", float("nan")),
                    stats.get("mr_p95", float("nan")),
                    stats.get("mr_p99", float("nan")),
                    stats.get("mr_min", float("nan")),
                    stats.get("mr_max", float("nan")),
                    stats.get("mr_mean", float("nan")),
                )
                logging.info(
                    "[CALIBRATION] Applied thresholds: LIVE_TRADE_EFF_THR_TREND=%.4f LIVE_TRADE_EFF_THR_MR=%.4f",
                    LIVE_TRADE_EFF_THR_TREND,
                    LIVE_TRADE_EFF_THR_MR,
                )
                logging.info("[CALIBRATION] Gate source: %s", stats.get("gate_source"))
                if CALIBRATION_SAVE_JSON:
                    payload = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "bars": CALIBRATION_BARS,
                        "percentile": CALIBRATION_PERCENTILE,
                        "use_full_dataset": CALIBRATION_USE_FULL_DATASET,
                        "trend_threshold": LIVE_TRADE_EFF_THR_TREND,
                        "mr_threshold": LIVE_TRADE_EFF_THR_MR,
                        "trend_stats": {k: stats.get(k) for k in stats if k.startswith("trend_")},
                        "mr_stats": {k: stats.get(k) for k in stats if k.startswith("mr_")},
                        "gate_source": stats.get("gate_source"),
                        "ckpt": str(ckpt),
                    }
                    try:
                        with open(CALIBRATION_JSON_PATH, "w", encoding="utf-8") as fh:
                            json.dump(payload, fh, indent=2)
                    except Exception as exc:
                        logging.warning("[CALIBRATION] Failed to write JSON: %s", exc)
        except Exception as exc:
            logging.warning("[CALIBRATION] Failed; using default thresholds. Error: %s", exc)
    usd_symbol = _select_usd_index_symbol()
    trader = TrendMRLiveTrader(model, feature_cols, norm_params, usd_symbol, dry_run, exit_confirm_bars)
    logging.info("Starting Trend/MR live loop (dry_run=%s, ckpt=%s)", dry_run, ckpt)
    logging.info("LIVE: entering trader.run()")
    trader.run()


def main_trend_mr() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training but is not available.")
    _ensure_output_dirs()
    config = _apply_overrides()
    logging.info("Applied OVERRIDE_GLOBALS keys: %s", ", ".join(config.get("applied_globals", [])))
    logging.info("Applied FEATURE_TOGGLES_BASE_OVERRIDE: %s", ", ".join(config.get("applied_toggles", [])))
    if config.get("warnings"):
        logging.info("Override keys ignored (not recognized): %s", ", ".join(config["warnings"]))
    effective = _effective_config()
    logging.info("Effective Config Summary:\n%s", json.dumps(effective, indent=2, default=str))
    logging.info("POS_WEIGHTS trade=%.2f dir=%.2f bounce=%.2f", POS_W_TRADE, POS_W_DIR, POS_W_BOUNCE)
    assert isinstance(RET_WINDOWS, (list, tuple)), "RET_WINDOWS must be list/tuple"
    assert isinstance(CORR_WINDOWS, (list, tuple)), "CORR_WINDOWS must be list/tuple"
    assert SHORT_LEN > 0 and LONG_LEN > 0, "Sequence lengths must be > 0"
    assert 0.0 < TRADE_THRESHOLD < 1.0, "TRADE_THRESHOLD must be in (0,1)"
    assert LEVEL_K >= 2, "LEVEL_K must be >= 2"
    current_lines = 0
    try:
        with open(__file__, "r", encoding="utf-8") as fh:
            current_lines = sum(1 for _ in fh)
    except Exception:
        current_lines = 0
    removed = max(0, ORIGINAL_LINE_COUNT - current_lines) if current_lines else 0
    logging.info(
        "Cleanup summary: removed ~%s lines; deleted RL update, parity harness, Stage B, and tick-level pipelines.",
        removed,
    )
    df, feature_cols = _build_feature_frame()
    _train_and_validate(df, feature_cols, config=config, effective_config=effective)
    if RUN_SMOKE_TEST:
        _smoke_test(df, feature_cols)


if __name__ == "__main__":
    try:
        print("BOOT: entering __main__", flush=True)
        parser = argparse.ArgumentParser(description="Trend/MR trader")
        parser.add_argument("--live", action="store_true", help="Run live MT5 loop")
        parser.add_argument("--dry_run", action="store_true", help="Run live loop without sending orders")
        parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for live mode")
        parser.add_argument(
            "--exit_confirm_bars",
            type=int,
            default=DIST_EXIT_CONFIRM_BARS,
            help="Consecutive closed bars to confirm exit",
        )
        parser.add_argument("--max_hold_minutes", type=int, default=None, help="Live-only max hold minutes before forced exit")
        parser.add_argument("--mt5_path", type=str, default=None, help="Path to terminal64.exe for MT5 routing")
        parser.add_argument("--diag_export", action="store_true", help="Enable diagnostic export mode")
        parser.add_argument("--diag_export_dir", type=str, default=DIAG_EXPORT_DIR, help="Diagnostic export directory")
        parser.add_argument("--diag_export_max_rows", type=int, default=DIAG_EXPORT_MAX_ROWS, help="Max diagnostic rows")
        parser.add_argument("--diag_export_split", type=str, default=DIAG_EXPORT_SPLIT, choices=["train", "val", "both"])
        parser.add_argument("--diag_export_every", type=int, default=DIAG_EXPORT_EVERY, help="Export every N epochs")
        parser.add_argument("--diag_export_format", type=str, default=DIAG_EXPORT_FORMAT, choices=["csv", "npz"])
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        logging.info("BOOT: parsed args: %s", args)
        if args.mt5_path:
            globals()["MT5_TERMINAL_PATH"] = args.mt5_path
        if args.diag_export:
            globals()["DIAG_EXPORT"] = True
        globals()["DIAG_EXPORT_DIR"] = args.diag_export_dir
        globals()["DIAG_EXPORT_MAX_ROWS"] = args.diag_export_max_rows
        globals()["DIAG_EXPORT_SPLIT"] = args.diag_export_split
        globals()["DIAG_EXPORT_EVERY"] = args.diag_export_every
        globals()["DIAG_EXPORT_FORMAT"] = args.diag_export_format
        if args.live:
            logging.info("about to enter run_live")
            run_live(
                dry_run=args.dry_run,
                ckpt_path=args.ckpt,
                exit_confirm_bars=args.exit_confirm_bars,
                max_hold_minutes=args.max_hold_minutes,
            )
        else:
            main_trend_mr()
            run_live(
                dry_run=args.dry_run,
                ckpt_path=args.ckpt,
                exit_confirm_bars=args.exit_confirm_bars,
                max_hold_minutes=args.max_hold_minutes,
            )
    except Exception:
        logging.exception("Run failed")
        raise
