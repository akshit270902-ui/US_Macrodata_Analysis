# btc-macro-sentiment

**Monthly US macro sentiment scoring for Bitcoin price modeling**, using a quantized Mistral-7B-Instruct model on Kaggle GPU.

Each month of US macro data (CPI, Fed Funds Rate, unemployment, payrolls, yield spread, M2) is scored on a **−1.0 to +1.0 scale** representing the net bearish/bullish macro impact on Bitcoin. Scores are then merged into a 1-hour BTC OHLCV dataset for downstream model training.

---

## What it does

1. **Loads and pivots** the FRED macro CSV into a monthly time-series of 6 key indicators.
2. **Computes month-over-month changes** for each indicator.
3. **Scores each month** via a prompted Mistral-7B-Instruct model (4-bit quantized via bitsandbytes), incorporating:
   - CPI MoM thresholds and YoY alert flags
   - Fed Funds Rate trend detection (hiking / pausing / cutting)
   - Pivot-candidate and dovish-cut boost signals
   - **Halving cycle context** — macro signal weights are attenuated based on the current phase (`early_bull`, `mid_bull`, `distribution`, `bear`, `pre_halving`) using only the public halving schedule (no price leakage)
4. **Caches scores** to CSV after each month; supports resuming interrupted runs.
5. **Merges** the monthly sentiment + halving features into a 1H BTC OHLCV DataFrame and saves the result.
6. **Runs diagnostics**:
   - Sanity checks against 5 known macro events (2020 COVID, 2022 hike start, 2022 inflation peak, 2023 pivot signal, 2024 rate cuts)
   - Parse quality report (full JSON vs. regex fallback vs. parse failures)
   - Phase-aware momentum drift check (detects streaks that contradict the cycle phase)
   - Score distribution health check (std, max, min, near-zero count)

---

## Inputs

| File | Description |
|---|---|
| `/kaggle/input/datasets/.../us_macro_history_*.csv` | FRED macro data, tab-separated, columns: `Date`, `Series ID`, `Value` |
| `/kaggle/input/datasets/.../BTCUSDT_1m.csv` | Binance 1-minute OHLCV |
| Local HF weights (optional) | Safetensors or `pytorch_model*.bin` under `/kaggle/input`; falls back to HuggingFace Hub download |

---

## Outputs

| File | Description |
|---|---|
| `/kaggle/working/macro_sentiment_scores.csv` | One row per month: `month`, `btc_macro_sentiment`, `reason`, `months_since_halving`, `cycle_phase`, `months_to_next_halving` |
| `/kaggle/working/BTCUSDT_1h_with_macro.csv` | 1H OHLCV with `btc_macro_sentiment`, `halving_cycle_phase`, `months_since_halving`, `months_to_next_halving` merged in |

---

## Halving cycle phases

Phase boundaries are based on publicly documented BTC supply-shock cycle patterns — no price data is used, so there is no lookahead leakage:

| Phase | Window | Macro weight |
|---|---|---|
| `early_bull` | 0–12 months post-halving | ~40% |
| `mid_bull` | 12–18 months post-halving | ~60% |
| `distribution` | 18–30 months post-halving | ~80% |
| `bear` | 30–48 months post-halving | ~90% |
| `pre_halving` | before first known halving | ~70% |

---

## Prompt design notes

The prompt includes several override rules that proved necessary to avoid known LLM failure modes:

- **Anti-compression rule**: prevents the model from defaulting to 0.0 on months with material data changes.
- **Scoring independence rule**: prevents score anchoring to the prior month.
- **Pivot-candidate hard floor**: when the Fed has held at cycle peak for 3+ months and CPI MoM < 0.50%, the score is forced ≥ 0.00. This is a `MANDATORY` constraint in the prompt to prevent CPI bearishness from overriding a clear pivot signal.
- **Hike-month guard**: the pivot-candidate label is suppressed entirely if the current month itself is a hike month, preventing early-2022 pre-hike holds from being misclassified.
- **YoY CPI alert flag**: injected into the prompt when YoY CPI ≥ 4%, so the model doesn't score a month as neutral/bullish based on a small MoM change while CPI is running at 6–9%.

---

## Setup

Designed to run in a **Kaggle notebook** with GPU access. bitsandbytes is installed at runtime.

```
# requirements (installed automatically)
bitsandbytes==0.45.5
transformers
torch
pandas
numpy
```

To use local model weights, upload a HuggingFace-compatible model to `/kaggle/input`; the script scans for `.safetensors` or `pytorch_model*.bin` and uses the first match. If none are found, it downloads from the Hub (requires Kaggle internet to be enabled).

---

## Score interpretation

| Score range | Macro regime |
|---|---|
| −1.0 to −0.40 | Strongly hawkish (aggressive hikes, high/sticky CPI) |
| −0.40 to −0.10 | Moderately hawkish |
| −0.10 to +0.10 | Neutral / mixed |
| +0.10 to +0.40 | Moderately dovish (cooling CPI, rising M2, minor cuts) |
| +0.40 to +1.0 | Strongly dovish (emergency cuts, surging M2, CPI collapse) |

---

## Known limitations

- The model has no knowledge of Fed *forward guidance* or FOMC meeting language — only the realized FRED data.
- Parse failures (model response not parseable) default to `0.0` and are flagged in the quality report.
- The `sparse_data_neutral` placeholder (score = 0.0) is saved for months with fewer than 2 non-NaN indicators; these are expected only at the very start of the series.
