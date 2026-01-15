# XAU-EDGE-TRADER-V1

A quantitative trading **research & execution showcase** for XAUUSD (Gold), built to demonstrate how modern ML systems can be structured for **signal generation, risk-aware decision making, and live execution**.

This repository focuses on:
- feature-driven neural models for **trend vs mean-reversion**
- **learned entry and learned exit policies**
- clean separation between **signal generation** and **trade control**
- realistic considerations for **live deployment** (logging, checkpoints, MT5 integration)

This is a **portfolio / showcase repo** intended for quant, systematic trading, and ML-for-finance roles.

---

## High-level architecture

The project is built around **two layers**:

### 1️⃣ Base Model (Signal + Risk Structure)
**`mr_sltp_edge_kelly.py`**

- Learns market structure (trend vs mean-reversion)
- Produces directional signals
- Learns auxiliary risk/edge representations
- Can run:
  - training / validation
  - dry-run live loop
  - real live execution via MT5

### 2️⃣ Meta-Policy (Decision Layer)
**`ensemble_meta_policy.py`**

- Treats one or more base models as **teachers**
- Trains:
  - an **ENTRY policy** → BUY / SELL / NO_TRADE
  - an **EXIT policy** → HOLD / EXIT
- Exits are **learned independently**, not based on signal flips
- Can train automatically if checkpoints are missing, then switch to live mode

This mirrors how real trading systems separate **forecasting** from **decision control**.

---

## Repository structure

```
XAU-EDGE-TRADER-V1/
│
├── mr_sltp_edge_kelly.py        # Base XAU model (train + live)
├── ensemble_meta_policy.py     # Meta entry/exit policy
├── Data/                        # Market data (CSV, exports, etc.)
├── Models/                      # Model checkpoints / configs
├── Outputs/                     # Logs, checkpoints, TensorBoard runs
├── LICENSE
└── README.md
```

---

## Requirements

### Hardware
- **CUDA-capable GPU required**
  - Both scripts explicitly refuse CPU-only execution

### Software
- Python **3.10+** recommended

Core dependencies:
- torch
- numpy
- pandas
- scikit-learn
- matplotlib
- tensorboard
- MetaTrader5 (only required for live trading)

---

## Setup (Step-by-step)

### Step 0 — Clone and create environment
```bash
git clone https://github.com/Rahul2304SP/XAU-EDGE-TRADER-V1.git
cd XAU-EDGE-TRADER-V1

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:
```bash
pip install torch numpy pandas scikit-learn matplotlib tensorboard MetaTrader5
```

> If you are **not running live**, MetaTrader5 is optional.

---

## Option A — Run the Base Model

### 1️⃣ Prepare data
The base model can load data from:
- MT5 directly, or
- CSV files (recommended for reproducibility)

Use **XAUUSD M1 data** with standard OHLC columns.

Place CSVs in your data directory (e.g. `Data/raw/`).

---

### 2️⃣ Train the model
Running without `--live` triggers training by default:
```bash
python mr_sltp_edge_kelly.py
```

This will:
- build features
- train and validate the model
- save checkpoints (`trend_mr_epoch*.pt`)
- log metrics

---

### 3️⃣ Run live (DRY RUN first)
```bash
python mr_sltp_edge_kelly.py --live --dry_run
```

To point to a specific MT5 terminal:
```bash
python mr_sltp_edge_kelly.py --live --dry_run --mt5_path "C:\Path\To\terminal64.exe"
```

Remove `--dry_run` only after demo testing.

---

## Option B — Run the Meta-Policy Ensemble (Recommended Showcase)

This is the **strongest demonstration** for quant roles: it shows signal abstraction, decision learning, and control separation.

---

### 1️⃣ Ensure base model checkpoints exist
Train at least one base model using `mr_sltp_edge_kelly.py`.

---

### 2️⃣ Train the ENTRY policy
```bash
python ensemble_meta_policy.py --train
```

This will:
- load base model outputs
- construct entry labels
- train BUY / SELL / NO_TRADE classifier
- save meta-policy checkpoints

---

### 3️⃣ Train the EXIT policy
```bash
python ensemble_meta_policy.py --train_exit_policy
```

This:
- generates exit labels from trade trajectories
- trains HOLD / EXIT policy
- saves exit policy checkpoint

---

### 4️⃣ Run live (DRY RUN first)
```bash
python ensemble_meta_policy.py --live --dry_run
```

If checkpoints are missing, the script will:
1. train them automatically
2. switch to live execution

Specify checkpoints manually if desired:
```bash
python ensemble_meta_policy.py --live --dry_run \
  --meta_ckpt Outputs/ensemble_meta/checkpoints/meta_epochXX.pt \
  --exit_policy_ckpt Outputs/ensemble_meta/checkpoints/exit_epochYY.pt
```

---

## Outputs & logging

### Base model outputs
- model checkpoints (`trend_mr_epoch*.pt`)
- training / validation metrics
- optional diagnostics

### Meta-policy outputs
```
Outputs/
└── ensemble_meta/
    ├── checkpoints/
    ├── logs/
    └── tb/          # TensorBoard runs
```

Launch TensorBoard:
```bash
tensorboard --logdir Outputs
```

---

## Why this repo is relevant for quant roles

This project demonstrates:

- separation of forecasting and decision-making
- risk-aware exits learned independently
- realistic execution loops (dry-run vs live)
- strong ML engineering hygiene (logging, checkpoints, reproducibility)

The architecture mirrors how real trading desks structure systematic strategies.

---

## Safety & disclaimer

This is **research code**.
- Use demo accounts first
- Apply strict risk limits
- Expect losses during experimentation

---

## Future extensions
- multi-symbol support
- portfolio-level risk constraints
- walk-forward validation tooling
- agentic / reinforcement-style policy refinement

---

## License
See `LICENSE` for details.
