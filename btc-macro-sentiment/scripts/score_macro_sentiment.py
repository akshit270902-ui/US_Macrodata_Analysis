"""
score_macro_sentiment.py
------------------------
Scores US macro monthly data for Bitcoin sentiment using a quantized Mistral-7B model.
Outputs a CSV of sentiment scores per month, then merges them into a 1H BTC OHLCV dataset.

Intended for use in a Kaggle notebook environment with GPU access.
"""

import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "--break-system-packages",
    "bitsandbytes==0.45.5"
])

import pandas as pd
import numpy as np
import torch
import json
import re
import os
import gc
import bitsandbytes
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ─────────────────────────────────────────────
# STEP 1 — Load & Pivot the macro CSV
# ─────────────────────────────────────────────

MACRO_PATH = '/kaggle/input/datasets/akshit2702/us-macro/us_macro_history_20260603_1318.csv'
START_DATE  = '2014-01-01'

macro_raw = pd.read_csv(MACRO_PATH, sep='\t')
if macro_raw.shape[1] < 4:
    macro_raw = pd.read_csv(MACRO_PATH)

macro_raw.columns = macro_raw.columns.str.strip()
macro_raw['Date'] = pd.to_datetime(macro_raw['Date'])
macro_raw = macro_raw[macro_raw['Date'] >= '2013-12-01'].copy()

macro_pivot = (
    macro_raw
    .pivot_table(index='Date', columns='Series ID', values='Value', aggfunc='last')
    .sort_index()
)

# ─────────────────────────────────────────────
# STEP 2 — Select the 6 key indicators
# ─────────────────────────────────────────────

KEY_INDICATORS = {
    'CPIAUCSL': 'CPI (All Urban Consumers)',
    'FEDFUNDS': 'Fed Funds Rate',
    'UNRATE':   'Unemployment Rate',
    'PAYEMS':   'Nonfarm Payrolls',
    'T10Y2Y':   '10Y-2Y Yield Spread',
    'M2SL':     'M2 Money Supply',
}

available = {k: v for k, v in KEY_INDICATORS.items() if k in macro_pivot.columns}
macro_key = macro_pivot[list(available.keys())].copy()

# ─────────────────────────────────────────────
# STEP 3 — Build month-over-month change table
# ─────────────────────────────────────────────

macro_prev = macro_key.shift(1)
macro_key  = macro_key[macro_key.index  >= START_DATE]
macro_prev = macro_prev[macro_prev.index >= START_DATE]

# ─────────────────────────────────────────────
# STEP 4 — Resolve model path
# ─────────────────────────────────────────────

HF_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
KERAS_ROOT  = "/kaggle/input"


def find_hf_weights_dir(root: str) -> str | None:
    root_path = Path(root)
    if not root_path.exists():
        return None
    for p in root_path.rglob("*.safetensors"):
        return str(p.parent)
    for p in root_path.rglob("pytorch_model*.bin"):
        return str(p.parent)
    return None


found_path = find_hf_weights_dir(KERAS_ROOT)
MODEL_PATH = found_path if found_path else HF_MODEL_ID

# ─────────────────────────────────────────────
# STEP 5 — Load tokenizer & model
# ─────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()

# ─────────────────────────────────────────────
# STEP 5b — Halving Cycle Encoder
# Uses only publicly known halving dates (no price data = no leakage).
# ─────────────────────────────────────────────

HALVING_DATES = [
    pd.Timestamp('2012-11-28'),
    pd.Timestamp('2016-07-09'),
    pd.Timestamp('2020-05-11'),
    pd.Timestamp('2024-04-20'),
    pd.Timestamp('2028-04-15'),  # estimated; based on public ~4-year cycle knowledge
]


def get_halving_features(date: pd.Timestamp) -> dict:
    """
    Returns 3 halving-cycle features for a given month.
    Uses ONLY the publicly known halving schedule — no price data, no leakage.

    Features
    --------
    months_since_halving    : int
    cycle_phase             : str — one of:
                                'early_bull'   (0–12 months post-halving)
                                'mid_bull'     (12–18 months post-halving)
                                'distribution' (18–30 months post-halving)
                                'bear'         (30–48 months post-halving)
                                'pre_halving'  (before first known halving)
    months_to_next_halving  : int  (-1 if unknown)
    """
    past_halvings   = [h for h in HALVING_DATES if h <= date]
    future_halvings = [h for h in HALVING_DATES if h  > date]

    months_to_next = int((future_halvings[0] - date).days / 30) if future_halvings else -1

    if not past_halvings:
        return {
            'months_since_halving':   -1,
            'cycle_phase':            'pre_halving',
            'months_to_next_halving': months_to_next,
        }

    last_halving = max(past_halvings)
    months_since = int((date - last_halving).days / 30)

    if months_since <= 12:
        phase = 'early_bull'
    elif months_since <= 18:
        phase = 'mid_bull'
    elif months_since <= 30:
        phase = 'distribution'
    else:
        phase = 'bear'

    return {
        'months_since_halving':   months_since,
        'cycle_phase':            phase,
        'months_to_next_halving': months_to_next,
    }


# ─────────────────────────────────────────────
# STEP 6 — Prompt builder & response parser
# ─────────────────────────────────────────────

SYSTEM_CONTEXT = """You are a macro analyst specializing in Bitcoin's sensitivity to US macroeconomic data.

Bitcoin price is POSITIVELY impacted by:
- Falling CPI (disinflation narrative → rate cuts expected)
- Fed Funds Rate being cut or expected to be cut
- Rising unemployment or slowing payrolls (forces Fed to pivot dovish)
- Yield curve un-inverting from deeply negative (recovery signal, liquidity incoming)
- Rising M2 money supply (more liquidity in system)
- Weak labor market + falling inflation together (strongest bull case)

Bitcoin price is NEGATIVELY impacted by:
- Persistent or rising CPI (delays rate cuts, dollar stays strong)
- Fed holding or hiking rates (tightening liquidity)
- Very strong payrolls/low unemployment (Fed has no reason to cut)
- Deeply negative or newly inverting yield curve (risk-off recession fear)
- Falling M2 (liquidity being drained)
- Mixed signals lean slightly negative (uncertainty = risk-off for BTC)

IMPORTANT NUANCES:
- Flat CPI at elevated levels (e.g. stuck at 3%+) = negative (no cut coming)
- Unemployment spiking sharply = short-term negative (panic) but medium-term positive (forces cut)
- Yield curve going from -0.5 to -0.1 (un-inverting) = cautiously positive
- Small changes near zero = neutral (0.0)
- Do NOT penalize low unemployment alone if the Fed has already paused hikes.
  Low unemployment only delays cuts; it is NOT the same as active tightening.

CPI MoM SCORING THRESHOLDS (apply these explicitly):
- MoM change < 0.10%          → bullish signal  (very low inflation, rate cuts likely)
- MoM change 0.10% – 0.25%    → neutral to mildly bullish
- MoM change 0.25% – 0.50%    → mildly bearish  (inflation sticky)
- MoM change 0.50% – 0.80%    → bearish         (hawkish pressure building)
- MoM change > 0.80%          → strongly bearish (rate hikes likely / cuts delayed)

PIVOT & PAUSE DETECTION RULES (override standard scoring when triggered):
- If the Fed Funds Rate trend over the last 3 months is FLAT or FALLING after a prior hiking
  cycle AND CPI MoM is below 0.40%, this is a PIVOT or PAUSE SIGNAL → score +0.15 to +0.35
  regardless of unemployment level. The hiking cycle ending is itself a bullish catalyst.
- If CPI MoM has been consistently declining for 2+ months (even if still above 2% YoY),
  treat as a disinflation narrative → lean bullish (+0.10 to +0.25).
- A Fed rate cut of any size after a prolonged hold = dovish catalyst → minimum +0.15 boost.

RATE CUT CYCLE SCORING (override CPI bias when an easing cycle is active):
- Once the Fed has begun an easing cycle (rate cuts confirmed in 2+ of the last 3 months),
  the baseline bias shifts to NEUTRAL-BULLISH regardless of CPI level.
  CPI above 2% during an active rate cut cycle means delayed future cuts, NOT new hikes.
  Score the DIRECTION CHANGE in monetary policy more heavily than the absolute CPI reading.
- The FIRST rate cut after a hold of 3+ months = minimum +0.15 boost, no exceptions.
- If Fed Funds Rate trend = "falling (easing / cutting cycle)" AND CPI MoM < 0.50%,
  the macro environment is net bullish for BTC. Do not score below -0.05 in this regime.
- A Fed pivot signal (market pricing in future cuts, Fed language shifting dovish) is itself
  a catalyst even before the first actual cut. Treat it as +0.10 to +0.20 bonus.

SCORING INDEPENDENCE RULE (critical):
- Score THIS month's data entirely on its own macroeconomic merits.
- The previous month's score is provided only as weak historical context.
- Do NOT anchor this month's score near the previous month's value.
- Each month must be evaluated independently.

SCORE RANGE CALIBRATION (use the full -1.0 to +1.0 scale):
- Strongly hawkish months (aggressive rate hikes, CPI MoM > 0.80%) → -0.50 to -0.70
- Moderately hawkish (sticky CPI, minor hikes) → -0.20 to -0.40
- Neutral / mixed → -0.10 to +0.10
- Moderately dovish (cooling CPI, rising M2, falling yield spreads, or minor dovish cuts) → +0.20 to +0.40
- Strongly dovish (emergency cuts, surging M2, CPI collapse) → +0.40 to +0.70
- Do NOT cluster scores near 0.00 unless the macro data is genuinely ambiguous.

ANTI-COMPRESSION RULE (mandatory):
- If CPI MoM > 0.30% OR fed_change > 0.10%, the score CANNOT be 0.00.
  A score of exactly 0.00 is reserved ONLY for genuinely flat, no-change months
  where ALL indicators moved less than 0.10%.
- When macro signals conflict (e.g. CPI up but Fed pausing), resolve the tension
  by picking the DOMINANT signal and scoring ±0.10 to ±0.25, not 0.00.
- "Mixed signals" → score -0.10 to -0.15 (slight negative bias for uncertainty),
  never exactly 0.00 unless data is truly flat across all indicators.
"""


def build_prompt(date, curr_row, prev_row, prev_score=None, rate_trend=None):
    changes_lines = []
    for col, label in available.items():
        curr_val = curr_row.get(col, np.nan)
        prev_val = prev_row.get(col, np.nan)
        if pd.isna(curr_val) or pd.isna(prev_val):
            changes_lines.append(f"- {label} ({col}): N/A")
        else:
            diff = curr_val - prev_val
            pct  = (diff / abs(prev_val) * 100) if prev_val != 0 else 0
            sign = "+" if diff >= 0 else ""
            changes_lines.append(
                f"- {label} ({col}): {prev_val:.3f} → {curr_val:.3f}  "
                f"({sign}{diff:.3f}, {sign}{pct:.2f}%)"
            )

    changes_text = "\n".join(changes_lines)
    month_str    = date.strftime("%B %Y")

    context_block = ""
    if prev_score is not None:
        context_block += (
            f"\nPrevious month macro sentiment score: {prev_score:+.2f} "
            f"(reference only — do NOT anchor this month's score near this value; "
            f"score this month's data independently on its own merits)"
        )
    if rate_trend is not None:
        context_block += f"\nFed Funds Rate trend over last 3 months: {rate_trend}"
    if context_block:
        context_block = f"\nCONTEXT FROM PRIOR MONTHS:{context_block}\n"

    # ── Halving cycle context ─────────────────────────────────────────────────
    halving = get_halving_features(date)

    phase_descriptions = {
        'early_bull': (
            "0–12 months post-halving. Supply shock is fresh and BTC is highly sensitive "
            "to both macro AND halving momentum simultaneously. Macro bearish signals may be "
            "partially offset by halving-driven demand. Weight macro at ~40% of final score — "
            "a moderately hawkish macro environment should push the score toward -0.10 to -0.25 "
            "rather than the full -0.40 to -0.65 it would warrant in a macro-dominant phase."
        ),
        'mid_bull': (
            "12–18 months post-halving. Peak sensitivity window where BOTH macro and halving "
            "cycle are simultaneously active and reinforcing. Weight macro at ~60%. "
            "A hawkish macro environment in this phase creates a genuine tug-of-war: "
            "score toward -0.15 to -0.35 rather than extreme bearish, unless macro is very severe."
        ),
        'distribution': (
            "18–30 months post-halving. Halving supply shock is largely priced in and fading. "
            "Macro becomes the primary driver. Weight macro at ~80%. "
            "Score macro signals closer to their face value, with only modest cycle adjustment."
        ),
        'bear': (
            "30–48 months post-halving. Halving effect has fully played out. BTC trades "
            "predominantly as a macro risk asset in this phase. Weight macro at ~90%. "
            "Score macro signals at near-full face value — cycle provides minimal offset."
        ),
        'pre_halving': (
            "Before first known halving or in accumulation phase approaching the next halving. "
            "Moderate macro sensitivity. Weight macro at ~70%."
        ),
    }

    phase_desc = phase_descriptions.get(halving['cycle_phase'], "Unknown phase.")

    halving_block = f"""
HALVING CYCLE CONTEXT (for {month_str}):
- Months since last halving  : {halving['months_since_halving']}
- Current cycle phase        : {halving['cycle_phase']}
- Months to next halving     : {halving['months_to_next_halving']}
- Phase meaning              : {phase_desc}

IMPORTANT: Do NOT use the cycle phase to blindly flip the score sign or ignore macro data.
Use it ONLY to calibrate how strongly macro signals dominate vs. being partially offset
by structural cycle momentum. The macro data always determines the DIRECTION; the cycle
phase adjusts the MAGNITUDE of bearish scores downward in early/mid bull phases.
"""

    # ── YoY CPI alert ─────────────────────────────────────────────────────────
    yoy_flag     = ""
    cpi_curr_val = curr_row.get('CPIAUCSL', np.nan)
    cpi_prev_val = prev_row.get('CPIAUCSL', np.nan)
    if not (pd.isna(cpi_curr_val) or pd.isna(cpi_prev_val)):
        date_12m_ago = date - pd.DateOffset(months=12)
        cpi_12m_ago  = (
            macro_key.loc[date_12m_ago, 'CPIAUCSL']
            if date_12m_ago in macro_key.index else np.nan
        )
        if not pd.isna(cpi_12m_ago) and cpi_12m_ago != 0:
            cpi_yoy = (cpi_curr_val - cpi_12m_ago) / cpi_12m_ago * 100
            if cpi_yoy >= 6.0:
                yoy_flag = (
                    f"\n🔴 HIGH INFLATION ALERT: CPI YoY = {cpi_yoy:.1f}%. "
                    "This is well above the Fed's 2% target and represents sustained, "
                    "broad-based inflation — NOT just a monthly blip. "
                    "Even if MoM CPI appears small this month, the YoY level means "
                    "rate cuts are firmly off the table and hikes are likely continuing. "
                    "Score must reflect this: minimum -0.40, likely -0.50 or worse, "
                    "regardless of cycle phase."
                )
            elif cpi_yoy >= 4.0:
                yoy_flag = (
                    f"\n🟡 ELEVATED INFLATION: CPI YoY = {cpi_yoy:.1f}%. "
                    "Inflation remains well above target. Rate cuts are not imminent. "
                    "Apply a bearish bias of at least -0.20 unless the Fed is actively cutting."
                )

    # ── Emergency / dovish cut flags ──────────────────────────────────────────
    stress_flag      = ""
    fedfunds_curr    = curr_row.get('FEDFUNDS', np.nan)
    fedfunds_prev_val = prev_row.get('FEDFUNDS', np.nan)
    if not (pd.isna(fedfunds_curr) or pd.isna(fedfunds_prev_val)):
        fed_drop = fedfunds_prev_val - fedfunds_curr
        if fed_drop >= 0.75 and fedfunds_prev_val <= 2.0:
            stress_flag = (
                f"\n⚠️  MACRO STRESS SIGNAL: Fed cut rates by {fed_drop:.2f}% in a single month "
                f"from an already-low base of {fedfunds_prev_val:.2f}%. "
                "This indicates an emergency policy response (e.g. financial crisis / pandemic). "
                "Apply a SHORT-TERM risk-off penalty (-0.10 to -0.25) to reflect panic conditions, "
                "but note the medium-term outlook is bullish due to extreme monetary easing. "
                "Final score should reflect the balance: net negative short-term, cautiously bullish longer-term."
            )
        elif fed_drop >= 0.25:
            stress_flag = (
                f"\n✅ DOVISH CATALYST: Fed cut rates by {fed_drop:.2f}% this month. "
                "This is a bullish signal for Bitcoin — looser monetary policy increases risk appetite "
                "and reduces the opportunity cost of holding non-yielding assets like BTC. "
                "Apply a minimum +0.15 boost for this dovish catalyst."
            )

    # ── Pause / pivot hint ────────────────────────────────────────────────────
    pause_hint = ""
    is_pivot_context = rate_trend is not None and (
        "pause" in rate_trend
        or "easing" in rate_trend
        or "pivot candidate" in rate_trend
    )
    if is_pivot_context and prev_score is not None and prev_score < -0.10:
        if "pivot candidate" in (rate_trend or ""):
            pause_hint = (
                f"\n📌 PIVOT CANDIDATE SIGNAL: The Fed has held rates flat at or near the "
                f"cycle peak for 3+ months (current rate trend: {rate_trend}). "
                "Markets historically price in future rate cuts 6–12 months before the "
                "first actual cut — this forward pricing IS a bullish catalyst for BTC "
                "even before cuts begin. "
                "\n⛔ HARD FLOOR — MANDATORY CONSTRAINT: If CPI MoM is below 0.50%, "
                "the score MUST be ≥ 0.00. This is not a suggestion. Do NOT override "
                "this floor with CPI bearishness — the pivot signal dominates when "
                "CPI MoM < 0.50%. A negative score in this regime is a scoring error. "
                "Target range: +0.10 to +0.30."
            )
        else:
            pause_hint = (
                "\n📌 REGIME CHANGE SIGNAL: The Fed Funds Rate has been flat or falling for 3 months "
                "after a sustained hiking cycle (evidenced by the deeply negative prior scores). "
                "This is a CONFIRMED PIVOT or PAUSE. The end of a hiking cycle is itself a major "
                "bullish catalyst for BTC regardless of current CPI level — markets price in future "
                "cuts, not current rates. "
                "\n⛔ HARD FLOOR — MANDATORY CONSTRAINT: The score MUST be ≥ 0.00 if CPI MoM "
                "is below 0.40%. This is not a guideline. Do NOT remain at the prior negative "
                "score level. A negative score here is a scoring error. "
                "Target range: +0.15 to +0.30."
            )

    prompt = f"""<s>[INST] {SYSTEM_CONTEXT}

Analyze the following US macro data changes for {month_str} and determine their combined impact on Bitcoin price.
{context_block}
{halving_block}
MACRO DATA CHANGES ({month_str}):
{changes_text}
{yoy_flag}{stress_flag}{pause_hint}

Respond ONLY with a valid JSON object. DO NOT include any text, explanation, or markdown before or after the JSON.
DO NOT truncate the response. The reply must end with a closing brace {{}}.
CRITICAL: Write your reasoning inside "reasoning" BEFORE choosing sentiment and score.

REASONING FORMAT: Write a concise 2-3 sentence assessment covering: (1) the CPI signal and its implication, (2) the Fed rate direction and what it means for liquidity, (3) the net combined BTC impact accounting for the current halving cycle phase.

NUMERICAL SCALING BOUNDS:
- -1.00 to -0.40: High/Sticky CPI, Fed rate hikes, falling M2, or deepening yield inversions.
- -0.39 to -0.10: Moderately hawkish indicators, strong jobs delaying pivots, or mixed data leaning negative.
- -0.09 to +0.09: Flat data, statistical noise, or perfectly balanced signals.
- +0.10 to +0.39: Cooling CPI, rising M2, falling yield spreads, or minor dovish cuts.
- +0.40 to +1.00: Aggressive Fed cuts, emergency injections, surging M2, or clear macro easing.

IMPORTANT: Use the FULL score range. Strong hawkish months (aggressive hikes, CPI MoM > 0.80%) MUST score -0.50 to -0.70. Strong dovish months (emergency cuts, CPI collapse) MUST score +0.40 to +0.70. Do NOT cluster near 0.
Remember: In early_bull or mid_bull cycle phases, moderate hawkish macro scores should be
attenuated (less negative) per the phase weighting described above.

Output exactly this JSON structure (no extra text before or after):
{{
  "reasoning": "Brief 2-3 sentence assessment: (1) CPI signal. (2) Fed rate direction. (3) Net BTC impact accounting for cycle phase.",
  "sentiment": "[bearish / neutral / bullish]",
  "score": [your_float_here]
}}
[/INST]"""
    return prompt


def parse_score(response_text: str) -> tuple[float, str]:
    """
    Six-attempt parse chain:
      0. Fast-path (score + sentiment found early)
      1. Full JSON parse
      2. Repaired JSON (truncated — append closing brace)
      3. Regex score extract (malformed but score field present)
      4. Sentiment keyword inference
      5. Return 0.0 / parse_failed
    """
    text = response_text.split("[/INST]")[-1].strip()

    # Attempt 0: fast-path
    early_score     = re.search(r'"score"\s*:\s*(-?\d+\.?\d*)', text)
    early_sentiment = re.search(r'"sentiment"\s*:\s*"(bearish|neutral|bullish)"', text, re.IGNORECASE)
    if early_score and early_sentiment:
        score   = float(early_score.group(1))
        keyword = early_sentiment.group(1).lower()
        if not (np.isnan(score) or np.isinf(score)) and -1.0 <= score <= 1.0:
            return round(score, 4), f"fast-path parse ({keyword})"

    # Attempt 1: full JSON
    try:
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            result    = json.loads(json_match.group())
            raw_score = result.get("score")
            reason    = result.get("reasoning", result.get("reason", "parsed successfully"))
            if raw_score is not None and isinstance(raw_score, (int, float)):
                score = float(raw_score)
                if not (np.isnan(score) or np.isinf(score)):
                    return round(float(np.clip(score, -1.0, 1.0)), 4), reason
    except Exception:
        pass

    # Attempt 2: repair truncated JSON
    try:
        partial = re.search(r'\{.+', text, re.DOTALL)
        if partial:
            raw      = partial.group().strip()
            raw      = re.sub(r',?\s*"[^"]*"\s*:\s*[^,}\n]*$', '', raw, flags=re.DOTALL)
            repaired = raw.rstrip(',\n ') + '}'
            result   = json.loads(repaired)
            raw_score = result.get("score")
            reason    = result.get("reasoning", result.get("reason", "repaired truncated JSON"))
            if raw_score is not None and isinstance(raw_score, (int, float)):
                score = float(raw_score)
                if not (np.isnan(score) or np.isinf(score)):
                    return round(float(np.clip(score, -1.0, 1.0)), 4), reason
    except Exception:
        pass

    # Attempt 3: regex score extraction
    score_match = re.search(r'"score"\s*:\s*(-?\d+\.?\d*)', text)
    if score_match:
        score = float(score_match.group(1))
        if not (np.isnan(score) or np.isinf(score)):
            return round(float(np.clip(score, -1.0, 1.0)), 4), "parsed via regex fallback"

    # Attempt 4: sentiment keyword
    sentiment_match = re.search(r'"sentiment"\s*:\s*"(bearish|neutral|bullish)"', text, re.IGNORECASE)
    if sentiment_match:
        keyword  = sentiment_match.group(1).lower()
        inferred = {"bearish": -0.25, "neutral": 0.0, "bullish": 0.25}[keyword]
        return inferred, f"inferred from sentiment keyword ({keyword})"

    return 0.0, "parse_failed"


def score_month(
    date,
    curr_row: dict,
    prev_row: dict,
    prev_score: float | None = None,
    rate_trend: str | None = None,
) -> tuple[float, str]:
    """Run LLM inference for a single month, retrying once on regex-fallback parse."""
    prompt = build_prompt(date, curr_row, prev_row, prev_score=prev_score, rate_trend=rate_trend)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    MAX_RETRIES    = 2
    score, reason  = 0.0, "parse_failed"

    for attempt in range(MAX_RETRIES + 1):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=700,
                temperature=max(0.02, 0.1 - attempt * 0.04),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        del outputs
        gc.collect()
        torch.cuda.empty_cache()

        score, reason = parse_score(response)
        if "regex fallback" in reason and attempt < MAX_RETRIES:
            continue
        break

    del inputs
    gc.collect()
    torch.cuda.empty_cache()

    return score, reason


# ─────────────────────────────────────────────
# STEP 7 — Run scoring loop over all months
# ─────────────────────────────────────────────

SCORES_CACHE = '/kaggle/working/macro_sentiment_scores.csv'
HALVING_COLS = ['months_since_halving', 'cycle_phase', 'months_to_next_halving']

if os.path.exists(SCORES_CACHE):
    scores_df = pd.read_csv(SCORES_CACHE, parse_dates=['month']).set_index('month')
    missing_cols = [c for c in HALVING_COLS if c not in scores_df.columns]
    if missing_cols:
        # Cache was written by an older version — re-score everything
        scores_df    = pd.DataFrame(columns=['month', 'btc_macro_sentiment', 'reason'] + HALVING_COLS).set_index('month')
        already_done = set()
    else:
        already_done = set(scores_df.index)
        print(f"Resuming — {len(already_done)} months already scored.")
else:
    scores_df    = pd.DataFrame(columns=['month', 'btc_macro_sentiment', 'reason'] + HALVING_COLS).set_index('month')
    already_done = set()

dates = macro_key.index.tolist()
total = len(dates)

for i, date in enumerate(dates):
    if date in already_done:
        continue

    curr_row = macro_key.loc[date].to_dict()
    prev_row = macro_prev.loc[date].to_dict()

    non_nan_curr = sum(1 for v in curr_row.values() if not pd.isna(v))
    non_nan_prev = sum(1 for v in prev_row.values() if not pd.isna(v))

    if non_nan_curr < 2 or non_nan_prev < 2:
        halving_feats = get_halving_features(date)
        scores_df.loc[date] = [
            0.0, "sparse_data_neutral",
            halving_feats['months_since_halving'],
            halving_feats['cycle_phase'],
            halving_feats['months_to_next_halving'],
        ]
        scores_df.reset_index().rename(columns={'index': 'month'}).to_csv(SCORES_CACHE, index=False)
        continue

    prev_score = float(scores_df.iloc[-1]['btc_macro_sentiment']) if len(scores_df) > 0 else None

    # ── Rate trend computation (with hike-month guard for pivot-candidate label) ──
    # The "pivot candidate" label is suppressed if the current month itself is a
    # hike month, preventing a pre-hike hold from being misclassified as a pivot.
    rate_trend = None
    if 'FEDFUNDS' in macro_key.columns:
        recent_dates = dates[max(0, i - 3):i]
        recent_fed = [
            macro_key.loc[d, 'FEDFUNDS']
            for d in recent_dates
            if not pd.isna(macro_key.loc[d, 'FEDFUNDS'])
        ]

        all_fed_up_to_now = [
            macro_key.loc[d, 'FEDFUNDS']
            for d in dates[:i]
            if not pd.isna(macro_key.loc[d, 'FEDFUNDS'])
        ]
        peak_fed     = max(all_fed_up_to_now) if all_fed_up_to_now else None
        curr_fed_val = macro_key.loc[date, 'FEDFUNDS'] if not pd.isna(macro_key.loc[date, 'FEDFUNDS']) else None

        if len(recent_fed) >= 2:
            if recent_fed[-1] > recent_fed[0] + 0.05:
                rate_trend = "rising (active hiking cycle)"
            elif recent_fed[-1] < recent_fed[0] - 0.05:
                rate_trend = "falling (easing / cutting cycle)"
            else:
                current_month_is_hike = (
                    len(recent_fed) >= 2 and recent_fed[-1] > recent_fed[-2] + 0.01
                )
                if (
                    not current_month_is_hike
                    and peak_fed
                    and curr_fed_val
                    and abs(curr_fed_val - peak_fed) < 0.10
                ):
                    months_at_peak = sum(
                        1 for d in dates[max(0, i - 6):i]
                        if not pd.isna(macro_key.loc[d, 'FEDFUNDS'])
                        and abs(macro_key.loc[d, 'FEDFUNDS'] - peak_fed) < 0.10
                    )
                    if months_at_peak >= 3:
                        rate_trend = (
                            f"flat at cycle peak ({curr_fed_val:.2f}% for "
                            f"{months_at_peak}+ months — pivot candidate)"
                        )
                    else:
                        rate_trend = "flat (pause / hold)"
                else:
                    rate_trend = "flat (pause / hold)"

    halving_feats = get_halving_features(date)
    score, reason = score_month(date, curr_row, prev_row, prev_score=prev_score, rate_trend=rate_trend)

    print(
        f"[{i+1}/{total}] {date.strftime('%Y-%m')} "
        f"[{halving_feats['cycle_phase']}] → {score:+.4f}"
    )

    scores_df.loc[date] = [
        score,
        reason,
        halving_feats['months_since_halving'],
        halving_feats['cycle_phase'],
        halving_feats['months_to_next_halving'],
    ]
    scores_df.reset_index().rename(columns={'index': 'month'}).to_csv(SCORES_CACHE, index=False)

print(f"✅ All {total} months scored → {SCORES_CACHE}")

# ─────────────────────────────────────────────
# STEP 7b — Gap check
# ─────────────────────────────────────────────

all_expected = pd.date_range(
    start=macro_key.index.min(),
    end=macro_key.index.max(),
    freq='MS'
)
scored_months = set(scores_df.index.normalize())
gaps = [m for m in all_expected if m not in scored_months]
if gaps:
    for m in gaps:
        print(f"  ⚠️  Missing: {m.strftime('%Y-%m')}")
    print(f"  {len(gaps)} month(s) missing — check the loop for skip conditions.")

# ─────────────────────────────────────────────
# STEP 8 — Load BTC 1-minute data & resample to 1H
# ─────────────────────────────────────────────

BTC_1M_PATH = '/kaggle/input/datasets/akshit2702/btcusd-g/BTCUSDT_1m.csv'

if not os.path.exists(BTC_1M_PATH):
    raise FileNotFoundError(f"BTC file not found: {BTC_1M_PATH}")

df = pd.read_csv(BTC_1M_PATH)
df.columns = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'count', 'taker_buy_volume',
    'taker_buy_quote_volume', 'ignore', 'source_date'
]
df['Gmt time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
df = df.set_index('Gmt time')

ohlc_dict = {
    'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'volume': 'sum', 'taker_buy_volume': 'sum',
}
df_1h = df.resample("1h").agg(ohlc_dict).dropna()
df_1h = df_1h[df_1h.index >= START_DATE]

# ─────────────────────────────────────────────
# STEP 9 — Merge macro sentiment + halving features
# ─────────────────────────────────────────────

df_1h['month_key'] = df_1h.index.to_period('M').to_timestamp()

df_1h['btc_macro_sentiment']    = df_1h['month_key'].map(scores_df['btc_macro_sentiment'].to_dict()).ffill()
df_1h['halving_cycle_phase']    = df_1h['month_key'].map(scores_df['cycle_phase'].to_dict()).ffill()
df_1h['months_since_halving']   = df_1h['month_key'].map(scores_df['months_since_halving'].to_dict()).ffill()
df_1h['months_to_next_halving'] = df_1h['month_key'].map(scores_df['months_to_next_halving'].to_dict()).ffill()
df_1h = df_1h.drop(columns=['month_key'])

# ─────────────────────────────────────────────
# STEP 10 — Save final output
# ─────────────────────────────────────────────

OUTPUT_PATH = '/kaggle/working/BTCUSDT_1h_with_macro.csv'
df_1h.to_csv(OUTPUT_PATH, index=True)

print(f"✅ Final dataset saved → {OUTPUT_PATH}  shape={df_1h.shape}")

# ─────────────────────────────────────────────
# STEP 11 — Sanity check: known macro events
# ─────────────────────────────────────────────

SANITY_CHECKS = {
    'COVID crash (2020-03)':        ('2020-03', 'mixed',    lambda s: abs(s) > 0.1),
    'Rate hike start (2022-03)':    ('2022-03', 'negative', lambda s: s < -0.20),
    'Inflation peak (2022-06)':     ('2022-06', 'negative', lambda s: s < -0.30),
    'Fed pivot signal (2023-11)':   ('2023-11', 'positive', lambda s: s > 0.0),
    'Rate cut start (2024-09)':     ('2024-09', 'positive', lambda s: s > 0.0),
}

print(f"\n{'Event':<38} {'Score':>7}   {'Phase':<14} {'Expected':>10}   Verdict")
print("─" * 100)
all_passed = True
for label, (ym, direction, check_fn) in SANITY_CHECKS.items():
    try:
        month_ts  = pd.Timestamp(ym + '-01')
        row       = scores_df.loc[month_ts]
        score_val = float(row['btc_macro_sentiment'])
        phase_val = str(row.get('cycle_phase', 'n/a'))
        verdict   = "✅ PASS" if check_fn(score_val) else "❌ FAIL"
        if "FAIL" in verdict:
            all_passed = False
        print(f"  {label:<36} {score_val:>+7.4f}   {phase_val:<14} {direction:>10}   {verdict}")
    except KeyError:
        print(f"  {label:<36} → not in scoring range")

print()
if all_passed:
    print("✅ All sanity checks passed.")
else:
    print(
        "⚠️  One or more sanity checks failed.\n"
        "   Review the prompt logic for the failing months — do NOT patch with forced overrides."
    )

# ─────────────────────────────────────────────
# STEP 12 — Parse quality report
# ─────────────────────────────────────────────

def _classify_reason(r: str) -> str:
    r = str(r)
    if 'parse_failed'           in r: return 'parse_failed'
    if 'sparse_data_neutral'    in r: return 'sparse_data_neutral'
    if 'regex fallback'         in r: return 'regex_fallback'
    if 'repaired'               in r: return 'repaired_json'
    if 'inferred from sentiment' in r: return 'sentiment_inferred'
    if 'fast-path parse'        in r: return 'fast_path_parse'
    return 'full_parse_ok'

reason_counts  = scores_df['reason'].apply(_classify_reason).value_counts()
total_months   = len(scores_df)
print("\nParse quality:")
for parse_type, count in reason_counts.items():
    flag = "⚠️ " if parse_type in ('parse_failed', 'regex_fallback', 'sentiment_inferred') else "✅ "
    print(f"  {flag}{parse_type:<25} {count:>4} months  ({count / total_months * 100:.1f}%)")

# ─────────────────────────────────────────────
# STEP 13 — Momentum drift check (phase-aware)
# ─────────────────────────────────────────────

DRIFT_WINDOW    = 4
DRIFT_THRESHOLD = 0.35
scores_series   = scores_df['btc_macro_sentiment'].sort_index()
drift_found     = False

for start_idx in range(len(scores_series) - DRIFT_WINDOW + 1):
    window        = scores_series.iloc[start_idx : start_idx + DRIFT_WINDOW]
    window_dates  = list(window.index)
    window_phases = [str(scores_df.loc[d, 'cycle_phase']) for d in window_dates]
    dominant      = max(set(window_phases), key=window_phases.count)

    if all(s <= -DRIFT_THRESHOLD for s in window) and dominant in ('early_bull', 'mid_bull'):
        months_str = [d.strftime('%Y-%m') for d in window_dates]
        print(
            f"  ⚠️  {DRIFT_WINDOW}-month BEARISH streak contradicts {dominant} phase: "
            f"{months_str} → {[f'{s:+.2f}' for s in window.values]}"
        )
        drift_found = True
    elif all(s >= DRIFT_THRESHOLD for s in window) and dominant in ('bear', 'distribution'):
        months_str = [d.strftime('%Y-%m') for d in window_dates]
        print(
            f"  ⚠️  {DRIFT_WINDOW}-month BULLISH streak contradicts {dominant} phase: "
            f"{months_str} → {[f'{s:+.2f}' for s in window.values]}"
        )
        drift_found = True

if not drift_found:
    print(f"✅ No suspicious momentum drift (no {DRIFT_WINDOW}-month streaks above ±{DRIFT_THRESHOLD} contradicting cycle phase).")

# ─────────────────────────────────────────────
# STEP 14 — Score distribution health check
# ─────────────────────────────────────────────

scores_clean = scores_df[scores_df['reason'] != 'sparse_data_neutral']['btc_macro_sentiment'].astype(float)
std_val      = scores_clean.std()
max_val      = scores_clean.max()
min_val      = scores_clean.min()
near_zero    = (scores_clean.abs() < 0.05).sum()
near_zero_pct = near_zero / len(scores_clean) * 100

print(
    f"\nScore distribution:"
    f"\n  Std  : {std_val:.4f}  {'✅' if std_val >= 0.20 else '⚠️  COLLAPSED — anchoring suspected'}"
    f"\n  Max  : {max_val:+.4f}  {'✅' if max_val >= 0.35 else '⚠️  LOW — model not outputting strong bullish scores'}"
    f"\n  Min  : {min_val:+.4f}  {'✅' if min_val <= -0.35 else '⚠️  HIGH — model not outputting strong bearish scores'}"
    f"\n  Near-zero (<0.05): {near_zero} months ({near_zero_pct:.1f}%) "
    f"{'⚠️  HIGH' if near_zero_pct > 20 else '✅'}"
)
