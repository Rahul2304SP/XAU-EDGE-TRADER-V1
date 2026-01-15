# XAU-EDGE-TRADER-V1
Gold forecasting showcase: MT5 data scraper, a meta-ensemble controller, and a 15m edge/Kelly SLTP model, plus MT5 trade-report analytics. Includes env-driven MT5 setup (no credentials), synthetic sample data for schema, and placeholder checkpoints—ready to demo how you ingest MT5 data, run an ensemble of teacher models, and analyze results.

# Gold Forecasting Showcase (Public Slice)

A trimmed, credential-free snapshot of the gold forecasting work: MT5 data scraping, a meta-policy ensemble, a mean-reversion SL/TP model, and MT5 trade-report analytics. All sensitive paths and credentials have been removed; bring your own checkpoints and MT5 access.

## Architecture at a glance
- Data ingestion: `mt5_scraper.py` pulls M1/ticks and writes symbol CSVs.
- Teacher models: `mr_sltp_edge_kelly.py` (and any others you plug in) emit `p_trade`/`p_dir` probabilities on their own feature pipelines.
- Meta-ensemble: `ensemble_meta_policy.py` consumes teacher outputs plus context features to decide BUY/SELL/NO_TRADE and learned EXIT.
- Evaluation: `mt5_trade_stats.py` parses MT5 reports for trade-level and bot-level metrics; plots go in `assets/`.

### Flow
```mermaid
flowchart LR
    A[MT5 Terminal\n(env-configured)] --> B[mt5_scraper.py\nM1 / ticks -> CSV]
    B --> C[Teacher Models\nmr_sltp_edge_kelly.py\n(checkpoints you provide)]
    C --> D[Meta-Policy Ensemble\nensemble_meta_policy.py\n(entry + exit decisions)]
    D --> E[MT5 Orders\n(live/dry-run)]
    E --> F[MT5 Reports\nXLSX exports]
    F --> G[mt5_trade_stats.py\ntrade/bot metrics]
    D --> H[Logs/Plots\nassets/]
```

## What''s inside
- `src/data/mt5_scraper.py`: MT5 M1/tick scraper with env-driven auth (no hardcoded keys).
- `src/models/ensemble_meta_policy.py`: meta-ensemble for entries/exits using teacher models (placeholder ckpts/paths).
- `src/models/mr_sltp_edge_kelly.py`: trend vs mean-reversion model with edge/kelly sizing logic (MT5 routing via env).
- `src/stats/mt5_trade_stats.py`: MT5 Excel report parser to per-trade and per-bot metrics.
- `data/sample_xau_m1.csv`: tiny synthetic M1 sample to demonstrate expected schema.
- `assets/`: drop your plots/equity curves before publishing.

## Quickstart
1) `python -m venv .venv && .venv/Scripts/Activate.ps1` (or use your env).
2) `pip install -r requirements.txt`
3) Copy `.env.example` to `.env` and fill MT5 values; or pass `--login/--password/--server` flags.
4) Kick the scraper on sample symbols (history only):
   `python src/data/mt5_scraper.py --mode m1 --symbols XAUUSD --oneshot`
5) Wire your checkpoints into `src/models/ensemble_meta_policy.py` (`DEFAULT_MODEL_SPECS`) and run dry-run inference:
   `python src/models/ensemble_meta_policy.py --live --dry_run`
6) Parse an MT5 report:
   `python src/stats/mt5_trade_stats.py path/to/ReportHistory.xlsx`

## Notes
- No model weights are shipped; point `ckpt_path` to your files before live use.
- MT5 terminal paths are env-driven to avoid leaking local installs.
- Sample data is synthetic and for structure only; replace with your exports.
- Add screenshots (e.g., feature importance, equity curve) to `assets/` before publishing.
