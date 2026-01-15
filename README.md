# XAU-EDGE-TRADER-V1 (Showcase Repo)

This repository is a **quant/ML trading systems showcase** built around:
1) a **base XAUUSD neural model** that learns **regime + entries + exits + sizing components**, and  
2) a **meta-policy ensemble** that learns **when to enter** (BUY/SELL/NO_TRADE) and **when to exit** (HOLD/EXIT) by consuming signals from one or more base models.

It’s designed to demonstrate:
- **end-to-end ML pipeline thinking** (feature engineering → supervised labels → training → evaluation → live execution)
- **risk-aware objectives** (drawdown penalties, tail-aware return heads, max hold constraints)
- **production-style engineering** (logging, checkpoints, deterministic outputs, MT5 integration)

---

## What’s in this repo

### 1) `mr_sltp_edge_kelly.py` — Base Model (Trend vs Mean Reversion)
A multi-head supervised model for XAUUSD that builds features from minute bars and learns:
- trend vs mean-reversion structure
- entry gating + direction
- multiple auxiliary heads (edge / quantile distribution / transition, etc.)
- dynamic SL/TP components and (optional) Kelly-style sizing logic

It can run in:
- **train mode** (default when you run it without `--live`)
- **live mode** via MT5 (`--live`)
- **dry run** (runs the live loop but does not place orders) (`--dry_run`) :contentReference[oaicite:2]{index=2}

### 2) `ensemble_meta_policy.py` — Meta-Policy Ensemble (Learned Entry + Learned Exit)
This script treats base models as **teachers** and trains two small policies:
- **Entry policy**: BUY / SELL / NO_TRADE (only acts when flat)
- **Exit policy**: HOLD / EXIT (only acts when in a position)

Key point: exits are learned and **do not depend on “signal flip”**. :contentReference[oaicite:3]{index=3}

It supports:
- `--train` (train entry policy; default mode)
- `--train_exit_policy` (train exit policy; will train entry policy first if missing)
- `--live` (run live; will auto-pick latest checkpoints or train if missing)
- `--dry_run` (recommended first) :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

---

## Requirements

### Hardware
- **CUDA GPU is required** (both scripts explicitly refuse CPU-only environments). :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

### Software
- Python 3.10+ recommended
- Key packages used in the code:
  - `torch`, `numpy`, `pandas`
  - `tensorboard`
  - `scikit-learn`
  - `matplotlib`
  - `MetaTrader5` (only required if you run live MT5 mode)

---

## Quickstart (Step-by-step)

### Step 0 — Clone and create a virtual environment
```bash
git clone https://github.com/Rahul2304SP/XAU-EDGE-TRADER-V1.git
cd XAU-EDGE-TRADER-V1

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
