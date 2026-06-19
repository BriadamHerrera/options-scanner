#!/usr/bin/env python3
"""
Professional Options Scanner — Anti-Fakeout Maximum Confluence
by Claude

Scoring (0–13 pts):
  1. Confirmed Donchian breakout (>0.5% clearance)  → +2pts
  2. EMA stack aligned (10>50>200 bulls / reverse bears) → +2pts  ← anti-fakeout
  3. ADX > threshold aligned with direction          → +1pt
  4. Bollinger Band squeeze                          → +1pt
  5. MACD histogram aligned                          → +1pt
  6. Volume spike > 1.5x avg                        → +1pt
  7. Strong closing candle (no rejection wick)       → +1pt  ← anti-fakeout
  8. Relative strength vs SPY (20-day)               → +1pt  ← anti-fakeout
  9. ATR not in spike (< 1.5x its 1yr avg)          → +1pt  ← anti-fakeout
 10. SPY market regime aligned                       → +1pt
 11. RSI not extreme                                 → +1pt
 12. IV Rank < 45%                                   → +0.5pt (via options)
 13. Tight bid/ask spread                            → +0.5pt (via options)
 14. No earnings within 5 days                       → +1pt

  Strong 9–13  |  Medium 6–8  |  Weak 3–5
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time

# ─── ALPACA REAL-TIME DATA ───────────────────────────────────────────────────────
ALPACA_KEY    = "PK37N65ATHZ2OTOR25L3U2WSXP"
ALPACA_SECRET = "BG15va2oxCwjMBKYKhA7a4ysqPGFfV93WVLaKXYqDNFq"

try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    _alpaca_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    ALPACA_OK = True
except Exception:
    ALPACA_OK = False

# ─── TIMEFRAME CONFIG ────────────────────────────────────────────────────────────
# Each entry: (label, alpaca_tf, yf_interval, lookback_days, cache_ttl_secs, donchian_period)
TF_OPTIONS = {
    "5min":  ("5min",  lambda: TimeFrame(5,  TimeFrameUnit.Minute), "5m",  5,   15,  48),
    "15min": ("15min", lambda: TimeFrame(15, TimeFrameUnit.Minute), "15m", 15,  30,  32),
    "1hr":   ("1hr",   lambda: TimeFrame.Hour,                      "1h",  45,  60,  20),
    "4hr":   ("4hr",   lambda: TimeFrame(4,  TimeFrameUnit.Hour),   "1h",  200, 120, 20),
    "Daily": ("Daily", lambda: TimeFrame.Day,                       "1d",  730, 300, 20),
}

# ─── WATCHLIST ───────────────────────────────────────────────────────────────────
# Robust names — profitable under BOTH the original AND optimized parameter sets
# (guards against curve-fitting). Chronic losers removed: TSLA, PLTR, NVDA, AVGO, META.
WATCHLIST = [
    "AMD","GOOGL","ARM","HOOD","QQQ",
    "MSTR","AMZN","COIN","SPY",
]
MIN_OI = 100  # raised from 50 — better liquidity

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Scanner Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size:1.5rem; font-weight:700; }
.anti-fakeout { background:#1a237e; color:#90caf9; padding:3px 9px; border-radius:4px; font-size:0.8rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Options Scanner Pro — Anti-Fakeout v3")
data_src = "🟢 Alpaca (real-time)" if ALPACA_OK else "🟡 yfinance (delayed)"
st.caption(f"Price data: {data_src}  ·  Options: yfinance  ·  13-factor anti-fakeout strategy  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# tf_label is set in sidebar — show active timeframe in a badge after sidebar renders

# ─── TRADING-STYLE PRESETS ───────────────────────────────────────────────────────
# Each preset sets the widget session_state keys before the widgets render.
PRESETS = {
    # Swing: Daily chart, 45 DTE (slow theta over 7-10d hold), score≥9, trend ADX
    "swing": {"k_tf": "Daily", "k_dte": 45, "k_score": 9, "k_iv": 50, "k_adx": 22,
              "k_spread": 20, "k_open": "Off"},
    # Same-day / intraday: 15min chart, short DTE, looser score, skip the open
    "day":   {"k_tf": "15min", "k_dte": 7,  "k_score": 6, "k_iv": 60, "k_adx": 18,
              "k_spread": 25, "k_open": "15 min"},
}

def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v

# Initialise widget defaults once (Swing-friendly) so presets can override cleanly.
_DEFAULTS = {"k_tf": "Daily", "k_dte": 30, "k_score": 6, "k_iv": 50,
             "k_adx": 20, "k_spread": 20, "k_open": "15 min"}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filters")

    strategy_label = st.radio(
        "📊 Strategy",
        ["📈 Trend / Breakout", "🔄 Mean-Reversion", "🎯 Supertrend Reversal", "🏔 Top-Down Levels"],
        index=0,
        help="Trend: buy breakouts (AMD, GOOGL). "
             "Mean-Reversion: fade extremes (PLTR, NVDA). "
             "Supertrend Reversal: ChartPrime-style + confluence. "
             "Top-Down Levels: key-level reclaim + higher-TF trend + 2:1 risk — the most "
             "robust directional setup in our tests (survived out-of-sample AND across "
             "universes), but ONLY on high-volatility stocks. Use 30 DTE. Paper-trade first.",
    )
    strategy_mode = ("mean_reversion" if "Mean" in strategy_label
                     else "chartprime_reversal" if "Supertrend" in strategy_label
                     else "topdown" if "Top-Down" in strategy_label
                     else "trend")

    st.caption("**Quick presets**")
    pc1, pc2 = st.columns(2)
    pc1.button("🎯 Swing (1–5d)", use_container_width=True,
               on_click=apply_preset, args=("swing",))
    pc2.button("⚡ Day Trade", use_container_width=True,
               on_click=apply_preset, args=("day",))

    tf_label = st.radio(
        "⏱ Timeframe",
        options=list(TF_OPTIONS.keys()),
        horizontal=True,
        key="k_tf",
    )
    tf_cfg = TF_OPTIONS[tf_label]

    custom_input = st.text_input("➕ Add tickers (comma-separated)", placeholder="UBER, ROKU, SNAP")
    watchlist = WATCHLIST.copy()
    if custom_input:
        watchlist += [t.strip().upper() for t in custom_input.split(",") if t.strip()]
        watchlist = list(dict.fromkeys(watchlist))

    target_dte     = st.select_slider("Target DTE", options=[7,14,21,30,45,60], key="k_dte")
    min_score      = st.slider("Min signal score (0–13)", 0, 13, key="k_score")
    max_iv_rank    = st.slider("Max IV Rank % (cheaper = lower)", 0, 100, key="k_iv")
    min_adx        = st.slider("Min ADX (trend strength)", 10, 40, key="k_adx")
    max_spread_pct = st.slider("Max bid/ask spread %", 5, 50, key="k_spread")

    st.divider()
    skip_earnings  = st.toggle("Skip stocks with earnings ≤5 days", value=True)
    show_wait      = st.toggle("Show 'WAIT' signals too", value=False)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    auto_refresh = st.toggle("Auto-refresh every 5 min", value=False)
    open_wait = st.selectbox(
        "⏳ Skip after market open",
        options=["Off", "15 min", "30 min", "60 min"],
        key="k_open",
        help="Ignores the chaotic opening candles after 9:30 ET on intraday timeframes — they produce the most fakeouts. Tune this once you see how it behaves live.",
    )
    open_wait_mins = {"Off": 0, "15 min": 15, "30 min": 30, "60 min": 60}[open_wait]

    st.divider()
    st.markdown("""
**Anti-Fakeout Filters** 🛡️
- 📐 **EMA Stack** — all 3 EMAs must line up (10>50>200)
- 🕯️ **Candle Quality** — no rejection wicks near the close
- 📊 **Rel. Strength** — stock must outperform SPY (calls) or underperform (puts)
- 📡 **ATR Regime** — avoid news spikes (ATR not abnormally high)
- ✅ **Confirmed Break** — must clear 20-day high/low by 0.5%+

**Score Thresholds**
- 🔥 9–13 → Strong
- ⚡ 6–8  → Medium
- 💤 3–5  → Weak
    """)


# ─── REAL-TIME BAR FETCHER ───────────────────────────────────────────────────────

def _drop_opening(df: pd.DataFrame, minutes: int = 15) -> pd.DataFrame:
    """Drop intraday bars within `minutes` of the 9:30 ET open (the choppiest, fakeout-prone candles)."""
    if df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        idx_et = idx.tz_localize("UTC").tz_convert("America/New_York")
    else:
        idx_et = idx.tz_convert("America/New_York")
    mins_since_open = (idx_et.hour - 9) * 60 + (idx_et.minute - 30)
    keep = ~((mins_since_open >= 0) & (mins_since_open < minutes))
    return df[keep]


@st.cache_data(ttl=15, show_spinner=False)
def fetch_bars(symbol: str, tf_label: str, open_wait_mins: int = 0) -> pd.DataFrame:
    """
    Fetch OHLCV bars for the chosen timeframe.
    Primary: Alpaca (real-time). Fallback: yfinance (delayed).
    """
    _, alpaca_tf_fn, yf_interval, lookback_days, _, _ = TF_OPTIONS[tf_label]
    # Opening filter only meaningful on intraday bars (not Daily)
    apply_open_filter = open_wait_mins > 0 and tf_label != "Daily"
    if ALPACA_OK:
        try:
            start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            req   = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_tf_fn(),
                start=start,
                feed="iex",
            )
            bars = _alpaca_client.get_stock_bars(req).df
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(symbol, level="symbol") if symbol in bars.index.get_level_values("symbol") else bars.droplevel(0)
            bars = bars.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
            bars = bars[["Open","High","Low","Close","Volume"]].copy()
            bars.index = pd.to_datetime(bars.index).tz_localize(None)
            bars.sort_index(inplace=True)
            if apply_open_filter:
                bars = _drop_opening(bars, open_wait_mins)
            if len(bars) >= 60:
                return bars
        except Exception:
            pass
    # Fallback: yfinance (4hr not supported natively — use 1h and resample)
    period_map = {5: "5d", 15: "15d", 45: "60d", 200: "1y", 730: "2y"}
    yf_period  = period_map.get(lookback_days, "60d")
    tk   = yf.Ticker(symbol)
    hist = tk.history(period=yf_period, interval=yf_interval)
    df   = hist[["Open","High","Low","Close","Volume"]].copy()
    # Resample to 4h if needed
    if tf_label == "4hr":
        df = df.resample("4h").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    if apply_open_filter:
        df = _drop_opening(df, open_wait_mins)
    return df


# ─── INDICATORS ──────────────────────────────────────────────────────────────────

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()


def confirmed_donchian(close, high, low, period=20, clearance=0.005):
    """
    Breakout must close above the 20-day high (or below low) by at least
    0.5% — filters out breakouts that just nick the level.
    """
    high20 = close.rolling(period).max().shift(1)
    low20  = close.rolling(period).min().shift(1)
    last_c = close.iloc[-1]
    last_h = high.iloc[-1]
    last_l = low.iloc[-1]

    if last_c > high20.iloc[-1] * (1 + clearance):
        return "BULLISH", 2
    elif last_c < low20.iloc[-1] * (1 - clearance):
        return "BEARISH", 2
    # Partial credit — touched but not confirmed
    elif last_c >= high20.iloc[-1]:
        return "BULLISH", 1
    elif last_c <= low20.iloc[-1]:
        return "BEARISH", 1
    return "NEUTRAL", 0


def ema_stack_score(close):
    """
    EMA alignment: 10 > 50 > 200 = fully bullish stack (+2)
    10 < 50 < 200 = fully bearish stack (+2)
    Partial alignment = +1
    """
    e10  = float(ema(close, 10).iloc[-1])
    e50  = float(ema(close, 50).iloc[-1])
    e200 = float(ema(close, 200).iloc[-1])
    last = float(close.iloc[-1])

    bull_full = last > e10 > e50 > e200
    bear_full = last < e10 < e50 < e200
    bull_part = last > e50 and e10 > e50
    bear_part = last < e50 and e10 < e50

    if bull_full:   return "BULLISH", 2
    if bear_full:   return "BEARISH", 2
    if bull_part:   return "BULLISH", 1
    if bear_part:   return "BEARISH", 1
    return "NEUTRAL", 0


def candle_quality(close, open_, high, low):
    """
    Anti-fakeout: the last candle should be a strong close with body > 50%
    of total range and little wick in the opposite direction.
    Bull candle: close near top of range, not a doji or bearish hammer.
    Bear candle: close near bottom of range.
    Returns (is_bull_quality, is_bear_quality).
    """
    c = float(close.iloc[-1])
    o = float(open_.iloc[-1])
    h = float(high.iloc[-1])
    l = float(low.iloc[-1])
    rng = h - l
    if rng == 0:
        return False, False
    body   = abs(c - o)
    body_pct = body / rng
    close_pct = (c - l) / rng  # 1.0 = close at high, 0.0 = close at low

    bull_ok = body_pct >= 0.4 and close_pct >= 0.6
    bear_ok = body_pct >= 0.4 and close_pct <= 0.4
    return bull_ok, bear_ok


def adx_calc(high, low, close, period=14):
    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    dm_plus  = (high - high.shift()).clip(lower=0)
    dm_minus = (low.shift() - low).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr
    dx       = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx_val  = dx.ewm(span=period, adjust=False).mean()
    return float(adx_val.iloc[-1]), float(di_plus.iloc[-1]), float(di_minus.iloc[-1])


def atr_regime(high, low, close, period=14):
    """
    Returns True if current ATR is NOT spiking (< 1.5x its 1-year average).
    Spikes = news events, gap moves — high fakeout risk.
    """
    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_pct  = (tr / close)
    current  = float(atr_pct.iloc[-1])
    avg_1yr  = float(atr_pct.rolling(252).mean().iloc[-1])
    if avg_1yr == 0:
        return True, 1.0
    ratio = round(current / avg_1yr, 2)
    return ratio < 1.5, ratio


def bb_squeeze(close, period=20):
    mid     = sma(close, period)
    std     = close.rolling(period).std()
    bw      = (std * 4) / mid
    bw_min  = bw.rolling(125).min()
    return bw.iloc[-1] <= bw_min.iloc[-1] * 1.05


def macd_signal(close):
    macd_line = ema(close, 12) - ema(close, 26)
    signal    = ema(macd_line, 9)
    hist      = macd_line - signal
    if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0: return "BULLISH"
    if hist.iloc[-1] < 0 and hist.iloc[-2] >= 0: return "BEARISH"
    if hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3]: return "BULLISH_TREND"
    if hist.iloc[-1] < hist.iloc[-2] < hist.iloc[-3]: return "BEARISH_TREND"
    return "NEUTRAL"


def rsi_calc(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def historical_vol(close, period=20):
    returns = np.log(close / close.shift()).dropna()
    return float(returns.rolling(period).std().iloc[-1] * np.sqrt(252) * 100)


# ─── MEAN-REVERSION SCORING ──────────────────────────────────────────────────────

def mr_score_at_bar(window):
    """Mean-reversion: buy oversold dips in uptrends / fade overbought rips. Score 0–10."""
    close, high, low, vol = window["Close"], window["High"], window["Low"], window["Volume"]
    c = float(close.iloc[-1])
    sma20 = float(sma(close,20).iloc[-1]); std20 = float(close.rolling(20).std().iloc[-1])
    sma200 = float(sma(close,200).iloc[-1]) if len(close) >= 200 else sma20
    if std20 == 0:
        return "NEUTRAL", 0, 50.0, 0.0
    z = (c - sma20)/std20
    rsi2  = float(rsi_calc(close, 2)); rsi14 = float(rsi_calc(close, 14))
    adx_v, _, _ = adx_calc(high, low, close)

    oversold   = rsi2 < 10 or z < -2
    overbought = rsi2 > 90 or z > 2
    if   oversold   and c > sma200: direction = "BULLISH"
    elif overbought and c < sma200: direction = "BEARISH"
    elif oversold:                  direction = "BULLISH"
    elif overbought:                direction = "BEARISH"
    else:
        return "NEUTRAL", 0, round(rsi2,1), round(adx_v,1)

    score = 0
    if (direction=="BULLISH" and rsi2<10) or (direction=="BEARISH" and rsi2>90): score += 2
    lower, upper = sma20-2.5*std20, sma20+2.5*std20
    if (direction=="BULLISH" and c<lower) or (direction=="BEARISH" and c>upper): score += 2
    if abs(z) >= 2: score += 1
    if adx_v < 25: score += 1
    if (direction=="BULLISH" and rsi14<30) or (direction=="BEARISH" and rsi14>70): score += 1
    diffs = close.diff().iloc[-3:]
    if (direction=="BULLISH" and (diffs<0).all()) or (direction=="BEARISH" and (diffs>0).all()): score += 1
    if (direction=="BULLISH" and c>sma200) or (direction=="BEARISH" and c<sma200): score += 1
    vavg = float(vol.iloc[-21:-1].mean()); vr = float(vol.iloc[-1])/vavg if vavg>0 else 1.0
    if vr >= 1.5: score += 1
    return direction, score, round(rsi2,1), round(adx_v,1)


def _topdown_signal_at_bar(window):
    """Top-down: 50-bar key-level reclaim aligned with 200-EMA trend, high-vol only.
       Returns (direction, stop_ref_price)."""
    close, high, low = window["Close"], window["High"], window["Low"]
    if len(close) < 200:
        return "NEUTRAL", None
    spot = float(close.iloc[-1]); o = float(window["Open"].iloc[-1])
    l = float(low.iloc[-1]); h = float(high.iloc[-1])
    e200 = float(ema(close, 200).iloc[-1])
    swing_lo = float(low.rolling(50).min().shift(1).iloc[-1])
    swing_hi = float(high.rolling(50).max().shift(1).iloc[-1])
    if historical_vol(close) < 40:      # high-vol gate (the strategy's requirement)
        return "NEUTRAL", None
    if spot > e200 and l < swing_lo and spot > swing_lo and spot > o:
        return "BULLISH", min(l, swing_lo)
    if spot < e200 and h > swing_hi and spot < swing_hi and spot < o:
        return "BEARISH", max(h, swing_hi)
    return "NEUTRAL", None


# ─── BACKTEST ENGINE ─────────────────────────────────────────────────────────────

def _score_at_bar(window, spy_window, min_adx_val):
    """Compute direction + score (0–13) at the last bar of `window`. Mirrors scan()."""
    close, high, low, open_, vol = window["Close"], window["High"], window["Low"], window["Open"], window["Volume"]
    score = 0
    direction, dpts = confirmed_donchian(close, high, low)
    score += dpts
    sdir, spts = ema_stack_score(close)
    if sdir == direction and spts > 0:
        score += spts
    adx_v, dip, dim = adx_calc(high, low, close)
    if adx_v >= min_adx_val and ((direction=="BULLISH" and dip>dim) or (direction=="BEARISH" and dim>dip)):
        score += 1
    if bb_squeeze(close):
        score += 1
    md = macd_signal(close)
    if (direction=="BULLISH" and md in ("BULLISH","BULLISH_TREND")) or (direction=="BEARISH" and md in ("BEARISH","BEARISH_TREND")):
        score += 1
    vavg = float(vol.iloc[-21:-1].mean()); vr = float(vol.iloc[-1])/vavg if vavg>0 else 1.0
    if vr >= 1.5:
        score += 1
    bc, brc = candle_quality(close, open_, high, low)
    if (direction=="BULLISH" and bc) or (direction=="BEARISH" and brc):
        score += 1
    sret = (float(close.iloc[-1])-float(close.iloc[-21]))/float(close.iloc[-21])*100
    spyret = (float(spy_window["Close"].iloc[-1])-float(spy_window["Close"].iloc[-21]))/float(spy_window["Close"].iloc[-21])*100
    if (direction=="BULLISH" and sret>spyret) or (direction=="BEARISH" and sret<spyret):
        score += 1
    aok, _ = atr_regime(high, low, close)
    if aok:
        score += 1
    spy_bull = float(spy_window["Close"].iloc[-1]) > float(ema(spy_window["Close"],50).iloc[-1])
    if (direction=="BULLISH" and spy_bull) or (direction=="BEARISH" and not spy_bull):
        score += 1
    rsi = rsi_calc(close)
    if (direction=="BULLISH" and rsi>=40) or (direction=="BEARISH" and rsi<=60):
        score += 1
    score += 1  # earnings assumed clear (not checked historically)
    return direction, score, round(adx_v,1), round(rsi,1)


@st.cache_data(ttl=3600, show_spinner=False)
def run_backtest(symbol, eval_days, min_score_bt, min_adx_val, hold_bars=10,
                 stop_pct=4.0, tgt_pct=12.0, trail_pct=4.0, trail_arm=4.0,
                 strategy="trend"):
    """
    Walk-forward backtest on the UNDERLYING. strategy = "trend" or "mean_reversion".
    Returns (trades_list, rows_for_table). Uses yfinance daily history.
    """
    try:
        data = yf.Ticker(symbol).history(period="2y")[["Open","High","Low","Close","Volume"]]
        spy  = yf.Ticker("SPY").history(period="2y")[["Open","High","Low","Close","Volume"]]
        data.index = data.index.tz_localize(None); spy.index = spy.index.tz_localize(None)
        common = data.index.intersection(spy.index)
        data, spy = data.loc[common], spy.loc[common]
    except Exception:
        return None, None
    if len(data) < 230:
        return None, None

    is_mr   = strategy == "mean_reversion"
    is_td   = strategy == "topdown"
    max_sc  = 10 if is_mr else 3 if is_td else 13
    mr_stop = 5.0   # mean-reversion knife-protection stop
    mr_hold = 7     # mean-reversion holds shorter
    start = len(data) - eval_days
    trades, table = [], []
    for i in range(max(start, 220), len(data)):
        w, sw = data.iloc[:i+1], spy.iloc[:i+1]
        td_stop_ref = None
        if is_mr:
            direction, score, rsi, adx_v = mr_score_at_bar(w)
        elif is_td:
            direction, td_stop_ref = _topdown_signal_at_bar(w)
            score = 2; rsi = round(float(rsi_calc(w["Close"])), 1); adx_v = 0.0
        else:
            direction, score, adx_v, rsi = _score_at_bar(w, sw, min_adx_val)
        floor = 0 if is_td else min_score_bt
        if direction == "NEUTRAL" or score < floor or i+1 >= len(data):
            continue
        entry = float(data["Open"].iloc[i+1]); exit_px, outcome, held = None, "TIME", 0
        peak = entry
        hb = mr_hold if is_mr else hold_bars
        if is_td:
            risk = (entry - td_stop_ref) if direction == "BULLISH" else (td_stop_ref - entry)
            if risk <= 0: risk = entry * 0.02
            td_stop = entry - risk if direction == "BULLISH" else entry + risk
            td_tgt  = entry + 2*risk if direction == "BULLISH" else entry - 2*risk
        for j in range(i+1, min(i+1+hb, len(data))):
            held += 1
            hi, lo, cl = float(data["High"].iloc[j]), float(data["Low"].iloc[j]), float(data["Close"].iloc[j])
            if is_mr:
                # Mean-reversion: exit on revert to SMA10, stop on further extension
                mean_now = float(sma(data["Close"].iloc[:j+1], 10).iloc[-1])
                if direction == "BULLISH":
                    if lo <= entry*(1-mr_stop/100): exit_px, outcome = entry*(1-mr_stop/100), "STOP"; break
                    if cl >= mean_now:              exit_px, outcome = cl, "MEAN"; break
                else:
                    if hi >= entry*(1+mr_stop/100): exit_px, outcome = entry*(1+mr_stop/100), "STOP"; break
                    if cl <= mean_now:              exit_px, outcome = cl, "MEAN"; break
            elif is_td:
                # Top-down: fixed 2:1 risk/reward on the underlying
                if direction == "BULLISH":
                    if lo <= td_stop: exit_px, outcome = td_stop, "STOP"; break
                    if hi >= td_tgt:  exit_px, outcome = td_tgt, "TARGET"; break
                else:
                    if hi >= td_stop: exit_px, outcome = td_stop, "STOP"; break
                    if lo <= td_tgt:  exit_px, outcome = td_tgt, "TARGET"; break
            else:
                if direction == "BULLISH":
                    if lo <= entry*(1-stop_pct/100): exit_px, outcome = entry*(1-stop_pct/100), "STOP"; break
                    peak = max(peak, hi)
                    if peak >= entry*(1+trail_arm/100) and lo <= peak*(1-trail_pct/100):
                        exit_px, outcome = peak*(1-trail_pct/100), "TRAIL"; break
                    if hi >= entry*(1+tgt_pct/100):  exit_px, outcome = entry*(1+tgt_pct/100), "TARGET"; break
                else:
                    if hi >= entry*(1+stop_pct/100): exit_px, outcome = entry*(1+stop_pct/100), "STOP"; break
                    peak = min(peak, lo)
                    if peak <= entry*(1-trail_arm/100) and hi >= peak*(1+trail_pct/100):
                        exit_px, outcome = peak*(1+trail_pct/100), "TRAIL"; break
                    if lo <= entry*(1-tgt_pct/100):  exit_px, outcome = entry*(1-tgt_pct/100), "TARGET"; break
        if exit_px is None:
            exit_px = float(data["Close"].iloc[min(i+hb, len(data)-1)])
        ret = (exit_px-entry)/entry*100 * (1 if direction=="BULLISH" else -1)
        trades.append(ret)
        table.append({
            "Date": str(data.index[i].date()),
            "Signal": "🟢 CALL" if direction=="BULLISH" else "🔴 PUT",
            "Score": f"{score}/{max_sc}", "ADX": adx_v, "RSI": rsi,
            "Entry": f"${entry:.2f}", "Exit": f"${exit_px:.2f}",
            "Outcome": outcome, "Days": held, "Return": f"{ret:+.2f}%",
        })
    return trades, table


def iv_rank_estimate(symbol, current_iv, tf_label, open_wait_mins=0):
    try:
        hist = fetch_bars(symbol, tf_label, open_wait_mins)
        if len(hist) < 60:
            return None
        returns   = np.log(hist["Close"] / hist["Close"].shift()).dropna()
        hv_series = returns.rolling(20).std() * np.sqrt(252) * 100
        hv_min    = hv_series.quantile(0.05)
        hv_max    = hv_series.quantile(0.95)
        if hv_max <= hv_min:
            return None
        return round(max(0, min(100, (current_iv - hv_min) / (hv_max - hv_min) * 100)), 1)
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def get_spy_regime(tf_label: str, open_wait_mins: int = 0):
    try:
        spy   = fetch_bars("SPY", tf_label, open_wait_mins)
        c     = spy["Close"]
        above = float(c.iloc[-1]) > float(ema(c, 50).iloc[-1])
        lookback = min(20, len(c) - 1)
        ret20 = (float(c.iloc[-1]) - float(c.iloc[-lookback])) / float(c.iloc[-lookback]) * 100
        return "BULL" if above else "BEAR", round(ret20, 2)
    except Exception:
        return "UNKNOWN", 0.0


@st.cache_data(ttl=30, show_spinner=False)
def get_spy_returns(tf_label: str, open_wait_mins: int = 0):
    try:
        spy     = fetch_bars("SPY", tf_label, open_wait_mins)
        c       = spy["Close"]
        lookback = min(20, len(c) - 1)
        return (float(c.iloc[-1]) - float(c.iloc[-lookback])) / float(c.iloc[-lookback]) * 100
    except Exception:
        return 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_date(symbol):
    try:
        cal = yf.Ticker(symbol).calendar
        if cal is None or cal.empty:
            return None
        if "Earnings Date" in cal.index:
            val = cal.loc["Earnings Date"]
            if hasattr(val, "__iter__") and not isinstance(val, str):
                val = list(val)[0]
            return pd.to_datetime(val).date()
    except Exception:
        pass
    return None


# ─── OPTIONS PICKER ──────────────────────────────────────────────────────────────

def best_contract(ticker_obj, direction, spot, target_dte):
    try:
        exps = ticker_obj.options
        if not exps:
            return None
        today = datetime.today().date()
        best_exp, best_diff = None, 999
        for exp in exps:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            if dte < 3:
                continue
            if abs(dte - target_dte) < best_diff:
                best_diff = abs(dte - target_dte)
                best_exp  = exp
        if not best_exp:
            return None

        dte   = (datetime.strptime(best_exp, "%Y-%m-%d").date() - today).days
        chain = ticker_obj.option_chain(best_exp)
        df    = chain.calls if direction == "BULLISH" else chain.puts
        df    = df[df["openInterest"] >= MIN_OI].copy()
        if df.empty:
            return None

        df["dist"] = (df["strike"] - spot).abs()
        pool = df[df["strike"] >= spot * 0.99] if direction == "BULLISH" else df[df["strike"] <= spot * 1.01]
        row  = (pool if not pool.empty else df).sort_values("dist").iloc[0]

        iv   = round(float(row.get("impliedVolatility", 0) or 0) * 100, 1)
        bid  = float(row.get("bid", 0) or 0)
        ask  = float(row.get("ask", 0) or 0)
        mid  = round((bid + ask) / 2, 2) if bid and ask else round(float(row.get("lastPrice", 0) or 0), 2)
        spread_pct = round((ask - bid) / mid * 100, 1) if mid > 0 else 999.0

        return {
            "type":       "CALL" if direction == "BULLISH" else "PUT",
            "expiry":     best_exp,
            "dte":        dte,
            "strike":     float(row["strike"]),
            "iv":         iv,
            "delta":      round(abs(float(row.get("delta", 0.5) or 0.5)), 2),
            "volume":     int(row.get("volume", 0) or 0),
            "oi":         int(row.get("openInterest", 0) or 0),
            "bid":        bid,
            "ask":        ask,
            "mid":        mid,
            "spread_pct": spread_pct,
        }
    except Exception:
        return None


# ─── SUPERTREND REVERSAL (ChartPrime-style + confluence filter) ──────────────────

def _supertrend_dir(hist, period=10, mult=2.0):
    """Supertrend direction series: +1 uptrend / -1 downtrend (ATR trailing stop)."""
    h, l, c = hist["High"], hist["Low"], hist["Close"]
    hl2 = (h + l) / 2
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    upper = hl2 + mult*atr; lower = hl2 - mult*atr
    fu = upper.copy(); fl = lower.copy()
    d = np.ones(len(hist))
    cv = c.values; uv = upper.values; lv = lower.values; fuv = fu.values; flv = fl.values
    for i in range(1, len(hist)):
        fuv[i] = min(uv[i], fuv[i-1]) if cv[i-1] <= fuv[i-1] else uv[i]
        flv[i] = max(lv[i], flv[i-1]) if cv[i-1] >= flv[i-1] else lv[i]
        if   cv[i] > fuv[i-1]: d[i] = 1
        elif cv[i] < flv[i-1]: d[i] = -1
        else:                  d[i] = d[i-1]
    return d


def _st_reversal_result(symbol, hist, target_dte, tf_label, open_wait_mins):
    """ChartPrime-style Supertrend reversal + 2-of-4 confluence filter."""
    close, high, low, volume = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    spot = float(close.iloc[-1]); prev = float(close.iloc[-2])
    chg  = round((spot - prev) / prev * 100, 2)

    d = _supertrend_dir(hist)
    flip_up = d[-1] == 1 and d[-2] == -1
    flip_dn = d[-1] == -1 and d[-2] == 1
    direction = "BULLISH" if flip_up else "BEARISH" if flip_dn else "NEUTRAL"

    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else spot
    rsi_v  = round(float(rsi_calc(close, 14)), 1)
    adx_v, _, _ = adx_calc(high, low, close); adx_v = round(adx_v, 1)
    vol_avg = float(volume.iloc[-21:-1].mean()); vol_ratio = round(float(volume.iloc[-1])/vol_avg, 2) if vol_avg>0 else 1.0

    conf = {}
    score = 0
    if direction != "NEUTRAL":
        bull = direction == "BULLISH"
        conf["Trend align (200MA)"] = (bull and spot > sma200) or (not bull and spot < sma200)
        conf["RSI confirms"]        = (bull and rsi_v < 45) or (not bull and rsi_v > 55)
        conf["Volume climax"]       = vol_ratio > 1.2
        conf["ADX < 35 (not extended)"] = adx_v < 35
        score = sum(1 for v in conf.values() if v)

    # require 2+ confluences (the validated filter)
    if direction == "NEUTRAL" or score < 2:
        signal = "WAIT"
    elif direction == "BULLISH":
        signal = "CALL"
    else:
        signal = "PUT"
    if score >= 4:   strength, sval = "🔥 Strong", 3
    elif score == 3: strength, sval = "⚡ Medium", 2
    else:            strength, sval = "💤 Weak", 1

    opt, iv_rank, spread_ok = None, None, False
    if signal != "WAIT":
        opt = best_contract(yf.Ticker(symbol), direction, spot, target_dte)
        if opt:
            iv_rank = iv_rank_estimate(symbol, opt["iv"], tf_label, open_wait_mins)
            spread_ok = opt["spread_pct"] < 15

    ed = get_earnings_date(symbol); today = datetime.today().date()
    dte_e = (ed - today).days if ed else None
    earn_near = dte_e is not None and 0 <= dte_e <= 5

    return {
        "symbol":symbol, "spot":spot, "chg":chg, "signal":signal, "direction":direction,
        "score":score, "score_max":4, "strategy":"chartprime_reversal",
        "strength":strength, "strength_val":sval, "adx":adx_v, "hv":0.0,
        "vol_ratio":vol_ratio, "opt":opt, "iv_rank":iv_rank, "spread_ok":spread_ok,
        "rsi_val":rsi_v, "rsi_ok":True, "conf_flags":conf,
        "breakout_pts":0, "confirmed":False, "ema_stack_pts":0, "ema_stack_full":False,
        "adx_ok":(adx_v<35), "squeeze":False, "macd_ok":False, "macd_dir":"NEUTRAL",
        "candle_ok":False, "rs_ok":False, "rs_val":0.0, "atr_ok":True, "atr_ratio":0.0,
        "spy_aligned":False, "earnings_date":str(ed) if ed else None,
        "days_to_earn":dte_e, "earnings_near":earn_near,
    }


# ─── TOP-DOWN KEY-LEVEL RECLAIM (level + reclaim + higher-TF trend + 2:1 risk) ───

def _topdown_result(symbol, hist, target_dte, tf_label, open_wait_mins):
    """Key 50-bar level reclaim, aligned with the 200-EMA trend. High-vol names only."""
    close, high, low, volume = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    spot = float(close.iloc[-1]); prev = float(close.iloc[-2])
    chg  = round((spot - prev) / prev * 100, 2)

    e200 = float(ema(close, 200).iloc[-1]) if len(close) >= 200 else spot
    swing_lo = float(low.rolling(50).min().shift(1).iloc[-1])
    swing_hi = float(high.rolling(50).max().shift(1).iloc[-1])
    o = float(hist["Open"].iloc[-1]); l = float(low.iloc[-1]); h = float(high.iloc[-1])
    trend_up = spot > e200

    direction = "NEUTRAL"; stop_ref = None
    # reclaim of support in an uptrend / reclaim of resistance in a downtrend
    if trend_up and l < swing_lo and spot > swing_lo and spot > o:
        direction = "BULLISH"; stop_ref = min(l, swing_lo)
    elif (not trend_up) and h > swing_hi and spot < swing_hi and spot < o:
        direction = "BEARISH"; stop_ref = max(h, swing_hi)

    # volatility gate — this setup only worked on high-vol underlyings
    hv = round(historical_vol(close), 1)
    vol_ok = hv >= 40  # annualized %; defensives/ETFs fall below this

    adx_v, _, _ = adx_calc(high, low, close); adx_v = round(adx_v, 1)
    rsi_v = round(float(rsi_calc(close)), 1)
    vol_avg = float(volume.iloc[-21:-1].mean()); vol_ratio = round(float(volume.iloc[-1])/vol_avg, 2) if vol_avg>0 else 1.0

    if direction == "NEUTRAL" or not vol_ok:
        signal = "WAIT"; score = 0
    else:
        score = 2 + (1 if vol_ratio >= 1.5 else 0)  # base 2, +1 for volume
        signal = "CALL" if direction == "BULLISH" else "PUT"
    strength, sval = ("🔥 Strong", 3) if score >= 3 else ("⚡ Medium", 2) if score == 2 else ("💤 Weak", 1)

    # risk levels (2:1) on the underlying
    risk_pct = round(abs(spot - stop_ref)/spot*100, 1) if stop_ref else None
    target_px = round(spot + 2*(spot-stop_ref), 2) if stop_ref and direction=="BULLISH" else round(spot - 2*(stop_ref-spot), 2) if stop_ref else None

    opt, iv_rank, spread_ok = None, None, False
    if signal != "WAIT":
        opt = best_contract(yf.Ticker(symbol), direction, spot, target_dte)
        if opt:
            iv_rank = iv_rank_estimate(symbol, opt["iv"], tf_label, open_wait_mins)
            spread_ok = opt["spread_pct"] < 15

    ed = get_earnings_date(symbol); today = datetime.today().date()
    dte_e = (ed - today).days if ed else None
    earn_near = dte_e is not None and 0 <= dte_e <= 5

    return {
        "symbol":symbol, "spot":spot, "chg":chg, "signal":signal, "direction":direction,
        "score":score, "score_max":3, "strategy":"topdown",
        "strength":strength, "strength_val":sval, "adx":adx_v, "hv":hv,
        "vol_ratio":vol_ratio, "opt":opt, "iv_rank":iv_rank, "spread_ok":spread_ok,
        "rsi_val":rsi_v, "rsi_ok":True, "td_stop_pct":risk_pct, "td_target":target_px,
        "td_stop_ref":round(stop_ref,2) if stop_ref else None, "td_vol_ok":vol_ok,
        "td_level":round(swing_lo if direction=="BULLISH" else swing_hi, 2) if direction!="NEUTRAL" else None,
        "breakout_pts":0, "confirmed":False, "ema_stack_pts":0, "ema_stack_full":False,
        "adx_ok":True, "squeeze":False, "macd_ok":False, "macd_dir":"NEUTRAL",
        "candle_ok":False, "rs_ok":False, "rs_val":0.0, "atr_ok":True, "atr_ratio":0.0,
        "spy_aligned":False, "earnings_date":str(ed) if ed else None,
        "days_to_earn":dte_e, "earnings_near":earn_near,
    }


# ─── CORE SCANNER ────────────────────────────────────────────────────────────────

def _mr_result(symbol, hist, target_dte, tf_label, open_wait_mins):
    """Build a render-compatible result dict for the Mean-Reversion strategy."""
    close, high, low, volume = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    spot = float(close.iloc[-1]); prev = float(close.iloc[-2])
    chg  = round((spot - prev) / prev * 100, 2)

    direction, score, rsi2, adx_v = mr_score_at_bar(hist)
    vol_avg = float(volume.iloc[-21:-1].mean()); vol_ratio = round(float(volume.iloc[-1])/vol_avg, 2) if vol_avg>0 else 1.0

    if direction == "NEUTRAL" or score < 4:
        signal = "WAIT"
    elif direction == "BULLISH":
        signal = "CALL"
    else:
        signal = "PUT"
    if score >= 7:   strength, sval = "🔥 Strong", 3
    elif score >= 5: strength, sval = "⚡ Medium", 2
    else:            strength, sval = "💤 Weak", 1

    opt, iv_rank, spread_ok = None, None, False
    if signal != "WAIT":
        opt = best_contract(yf.Ticker(symbol), direction, spot, target_dte)
        if opt:
            iv_rank = iv_rank_estimate(symbol, opt["iv"], tf_label, open_wait_mins)
            spread_ok = opt["spread_pct"] < 15

    # earnings (still relevant — avoid IV crush)
    ed = get_earnings_date(symbol); today = datetime.today().date()
    dte_e = (ed - today).days if ed else None
    earn_near = dte_e is not None and 0 <= dte_e <= 5

    return {
        "symbol":symbol, "spot":spot, "chg":chg, "signal":signal, "direction":direction,
        "score":score, "score_max":10, "strategy":"mean_reversion",
        "strength":strength, "strength_val":sval, "adx":adx_v, "hv":0.0,
        "vol_ratio":vol_ratio, "opt":opt, "iv_rank":iv_rank, "spread_ok":spread_ok,
        "rsi2":rsi2, "rsi_val":rsi2, "rsi_ok":True,
        # trend-specific keys → neutral defaults so renderers don't crash
        "breakout_pts":0, "confirmed":False, "ema_stack_pts":0, "ema_stack_full":False,
        "adx_ok":(adx_v<25), "squeeze":False, "macd_ok":False, "macd_dir":"NEUTRAL",
        "candle_ok":False, "rs_ok":False, "rs_val":0.0, "atr_ok":True, "atr_ratio":0.0,
        "spy_aligned":False, "earnings_date":str(ed) if ed else None,
        "days_to_earn":dte_e, "earnings_near":earn_near,
    }


@st.cache_data(ttl=15, show_spinner=False)
def scan(symbol, target_dte, min_adx_val, spy_regime, spy_ret20, tf_label, open_wait_mins=0, strategy="trend"):
    try:
        _, _, _, _, _, don_period = TF_OPTIONS[tf_label]
        hist = fetch_bars(symbol, tf_label, open_wait_mins)
        if len(hist) < 60:  # minimum bars needed
            return None

        if strategy == "mean_reversion":
            return _mr_result(symbol, hist, target_dte, tf_label, open_wait_mins)
        if strategy == "chartprime_reversal":
            return _st_reversal_result(symbol, hist, target_dte, tf_label, open_wait_mins)
        if strategy == "topdown":
            return _topdown_result(symbol, hist, target_dte, tf_label, open_wait_mins)

        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        open_  = hist["Open"]
        volume = hist["Volume"]
        spot   = float(close.iloc[-1])
        prev   = float(close.iloc[-2])
        chg    = round((spot - prev) / prev * 100, 2)

        score = 0
        details = {}

        # ── 1. Confirmed Donchian breakout (0–2 pts) ─────────────────────────
        direction, don_score = confirmed_donchian(close, high, low, period=don_period)
        score += don_score
        details["breakout_pts"] = don_score
        details["confirmed"]    = don_score == 2  # True = cleared 0.5%

        # ── 2. EMA stack alignment (0–2 pts) [ANTI-FAKEOUT] ──────────────────
        stack_dir, stack_pts = ema_stack_score(close)
        ema_aligned = stack_dir == direction and stack_pts > 0
        if ema_aligned:
            score += stack_pts
        details["ema_stack_pts"] = stack_pts if ema_aligned else 0
        details["ema_stack_full"] = stack_pts == 2 and ema_aligned

        # ── 3. ADX trend strength (0–1 pt) ───────────────────────────────────
        adx_val, di_plus, di_minus = adx_calc(high, low, close)
        adx_val = round(adx_val, 1)
        adx_ok = adx_val >= min_adx_val and (
            (direction == "BULLISH" and di_plus > di_minus) or
            (direction == "BEARISH" and di_minus > di_plus)
        )
        if adx_ok:
            score += 1
        details["adx_ok"] = adx_ok

        # ── 4. BB squeeze (0–1 pt) ────────────────────────────────────────────
        squeeze = bb_squeeze(close)
        if squeeze:
            score += 1
        details["squeeze"] = squeeze

        # ── 5. MACD (0–1 pt) ──────────────────────────────────────────────────
        macd_dir = macd_signal(close)
        macd_ok  = (
            (direction == "BULLISH" and macd_dir in ("BULLISH", "BULLISH_TREND")) or
            (direction == "BEARISH" and macd_dir in ("BEARISH", "BEARISH_TREND"))
        )
        if macd_ok:
            score += 1
        details["macd_ok"] = macd_ok
        details["macd_dir"] = macd_dir

        # ── 6. Volume spike (0–1 pt) ──────────────────────────────────────────
        vol_avg   = float(volume.iloc[-21:-1].mean())
        vol_today = float(volume.iloc[-1])
        vol_ratio = round(vol_today / vol_avg, 2) if vol_avg > 0 else 1.0
        if vol_ratio >= 1.5:
            score += 1
        details["vol_ratio"] = vol_ratio

        # ── 7. Strong candle quality (0–1 pt) [ANTI-FAKEOUT] ─────────────────
        bull_candle, bear_candle = candle_quality(close, open_, high, low)
        candle_ok = (direction == "BULLISH" and bull_candle) or (direction == "BEARISH" and bear_candle)
        if candle_ok:
            score += 1
        details["candle_ok"] = candle_ok

        # ── 8. Relative strength vs SPY (0–1 pt) [ANTI-FAKEOUT] ──────────────
        rs_ok = False
        rs_val = 0.0
        if symbol != "SPY" and len(close) >= 21:
            stock_ret20 = (float(close.iloc[-1]) - float(close.iloc[-21])) / float(close.iloc[-21]) * 100
            rs_val      = round(stock_ret20 - spy_ret20, 2)
            if direction == "BULLISH" and stock_ret20 > spy_ret20:
                rs_ok = True
                score += 1
            elif direction == "BEARISH" and stock_ret20 < spy_ret20:
                rs_ok = True
                score += 1
        details["rs_ok"]  = rs_ok
        details["rs_val"] = rs_val

        # ── 9. ATR regime — not spiking (0–1 pt) [ANTI-FAKEOUT] ─────────────
        atr_ok, atr_ratio = atr_regime(high, low, close)
        if atr_ok:
            score += 1
        details["atr_ok"]    = atr_ok
        details["atr_ratio"] = atr_ratio

        # ── 10. SPY market regime (0–1 pt) ───────────────────────────────────
        spy_aligned = False
        if symbol != "SPY":
            if direction == "BULLISH" and spy_regime == "BULL":
                spy_aligned = True
                score += 1
            elif direction == "BEARISH" and spy_regime == "BEAR":
                spy_aligned = True
                score += 1
        else:
            spy_aligned = True
        details["spy_aligned"] = spy_aligned

        # ── 11. RSI momentum (0–1 pt) ─────────────────────────────────────────
        # No upper cap for calls / lower cap for puts — momentum continuation
        # outperformed in backtest (RSI 70+ was the best bucket, not the worst).
        rsi_val = round(rsi_calc(close), 1)
        rsi_ok  = (
            (direction == "BULLISH" and rsi_val >= 40) or
            (direction == "BEARISH" and rsi_val <= 60)
        )
        if rsi_ok:
            score += 1
        details["rsi_ok"]  = rsi_ok
        details["rsi_val"] = rsi_val

        # ── 12. Earnings proximity (0–1 pt) ──────────────────────────────────
        earnings_date = get_earnings_date(symbol)
        today         = datetime.today().date()
        days_to_earn  = (earnings_date - today).days if earnings_date else None
        earnings_near = days_to_earn is not None and 0 <= days_to_earn <= 5
        if not earnings_near:
            score += 1
        details["earnings_date"] = str(earnings_date) if earnings_date else None
        details["days_to_earn"]  = days_to_earn
        details["earnings_near"] = earnings_near

        hv_val = round(historical_vol(close), 1)

        # Signal
        if direction == "NEUTRAL" or score < 3:
            signal = "WAIT"
        elif direction == "BULLISH":
            signal = "CALL"
        else:
            signal = "PUT"

        if score >= 9:
            strength, strength_val = "🔥 Strong", 3
        elif score >= 6:
            strength, strength_val = "⚡ Medium", 2
        else:
            strength, strength_val = "💤 Weak", 1

        # ── 13+14. Options: IV rank + tight spread ────────────────────────────
        opt       = None
        iv_rank   = None
        spread_ok = False

        if signal != "WAIT":
            yf_tk = yf.Ticker(symbol)  # yfinance only for options chain
            opt = best_contract(yf_tk, direction, spot, target_dte)
            if opt:
                iv_rank = iv_rank_estimate(symbol, opt["iv"], tf_label, open_wait_mins)
                if iv_rank is not None and iv_rank <= 45:
                    score += 1
                if opt["spread_pct"] < 15:
                    spread_ok = True
                    score += 1
                # Recalculate strength after options pts
                if score >= 9:
                    strength, strength_val = "🔥 Strong", 3
                elif score >= 6:
                    strength, strength_val = "⚡ Medium", 2

        return {
            "symbol":        symbol,
            "spot":          spot,
            "chg":           chg,
            "signal":        signal,
            "direction":     direction,
            "score":         score,
            "score_max":     13,
            "strategy":      "trend",
            "strength":      strength,
            "strength_val":  strength_val,
            "adx":           adx_val,
            "hv":            hv_val,
            "vol_ratio":     vol_ratio,
            "opt":           opt,
            "iv_rank":       iv_rank,
            "spread_ok":     spread_ok,
            **details,
        }
    except Exception:
        return None


# ─── SCAN ────────────────────────────────────────────────────────────────────────

spy_regime, spy_mom = get_spy_regime(tf_label, open_wait_mins)
spy_ret20           = get_spy_returns(tf_label, open_wait_mins)
spy_label = (
    f"🐂 Bull Market (SPY +{spy_mom:.1f}% / 20d)" if spy_regime == "BULL"
    else f"🐻 Bear Market (SPY {spy_mom:.1f}% / 20d)" if spy_regime == "BEAR"
    else "❓ Unknown Regime"
)
_open_note = f"  ·  ⏳ skipping first {open_wait_mins} min" if (open_wait_mins and tf_label != "Daily") else ""
_strat_badge = ("🔄 **Mean-Reversion**" if strategy_mode == "mean_reversion"
                else "🎯 **Supertrend Reversal**" if strategy_mode == "chartprime_reversal"
                else "🏔 **Top-Down Levels**" if strategy_mode == "topdown"
                else "📈 **Trend / Breakout**")
st.info(f"**Strategy:** {_strat_badge}  |  **Market Regime:** {spy_label}  |  ⏱ **{tf_label}**{_open_note}")
if strategy_mode == "topdown":
    st.warning("🏔 **Top-Down Levels** — fires when price reclaims a 50-bar key level *in the direction of the 200-EMA trend*, on **high-volatility names only** (HV ≥ 40%; defensives/ETFs are filtered out). Score 0–3, set 'Min signal score' to **2**. **Use 30 DTE**, risk via the 2:1 stop/target shown. ✅ Most robust setup in our tests (survived out-of-sample *and* across stock universes) — but still **regime-dependent** (recent year was negative). Paper-trade first.")
if strategy_mode == "mean_reversion":
    st.caption("🔄 Mean-Reversion fades oversold dips / overbought rips. Best on choppy names (PLTR, NVDA). Score is out of 10. Lower your 'Min signal score' to ~5–6 since MR maxes at 10.")
if strategy_mode == "chartprime_reversal":
    st.warning("🎯 **Supertrend Reversal** fires only when the ATR trend flips AND ≥2 of 4 confluences confirm. Score is out of 4 — set 'Min signal score' to **2**. **Use 30 DTE.** ⚠️ This was the best directional setup in backtests but is **regime-dependent (profitable ~2 of 4 years)** — paper-trade it, size small, expect losing stretches. Signals are rare by design (only on fresh flips).")

bar  = st.progress(0, text="Scanning…")
rows = []
for i, sym in enumerate(watchlist):
    bar.progress((i + 1) / len(watchlist), text=f"Scanning {sym}…")
    r = scan(sym, target_dte, min_adx, spy_regime, spy_ret20, tf_label, open_wait_mins, strategy_mode)
    if r is None:
        continue
    # cap the score floor to each strategy's own scale so signals aren't silently hidden
    eff_floor = min(min_score, r.get("score_max", 13))
    if strategy_mode == "chartprime_reversal":
        eff_floor = min(min_score, 2)  # 2-confluence is the validated filter
    if strategy_mode == "topdown":
        eff_floor = min(min_score, 2)  # base setup scores 2
    if r["score"] < eff_floor:
        continue
    if not show_wait and r["signal"] == "WAIT":
        continue
    if r["opt"] is not None and r["iv_rank"] is not None and r["iv_rank"] > max_iv_rank:
        continue
    if skip_earnings and r["earnings_near"]:
        continue
    if r["opt"] is not None and r["opt"]["spread_pct"] > max_spread_pct:
        continue
    rows.append(r)

bar.empty()
rows.sort(key=lambda r: (-r["score"], -abs(r["chg"])))

calls = [r for r in rows if r["signal"] == "CALL"]
puts  = [r for r in rows if r["signal"] == "PUT"]
waits = [r for r in rows if r["signal"] == "WAIT"]

# ─── SUMMARY METRICS ─────────────────────────────────────────────────────────────

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🟢 Call Signals",    len(calls))
m2.metric("🔴 Put Signals",     len(puts))
m3.metric("⚪ Wait",            len(waits))
m4.metric("🔥 Strong Signals",  sum(1 for r in rows if r["strength_val"] == 3))
m5.metric("📋 Total Scanned",   len(rows))

st.divider()

# ─── SIGNAL TABLE ────────────────────────────────────────────────────────────────

def render_table(data):
    if not data:
        st.info("No signals match current filters.")
        return

    table = []
    for r in data:
        opt = r["opt"]
        sig = ("🟢 CALL" if r["signal"] == "CALL"
               else "🔴 PUT" if r["signal"] == "PUT"
               else "⚪ WAIT")

        flags = []
        if r.get("strategy") == "mean_reversion":
            flags.append("🔄 MR")
            if r.get("rsi2") is not None:    flags.append(f"RSI2 {r['rsi2']:.0f}")
            if r.get("adx_ok"):              flags.append(f"〰 Choppy(ADX{r['adx']})")
        elif r.get("strategy") == "chartprime_reversal":
            flags.append("🎯 Reversal")
            for lab, ok in r.get("conf_flags", {}).items():
                if ok: flags.append(f"✓{lab.split()[0]}")
        elif r.get("strategy") == "topdown":
            flags.append("🏔 Reclaim")
            if r.get("td_level") is not None: flags.append(f"lvl ${r['td_level']}")
            if r.get("hv"): flags.append(f"HV{r['hv']:.0f}")
            if r.get("td_stop_pct"): flags.append(f"risk {r['td_stop_pct']}%")
        else:
            if r.get("confirmed"):           flags.append("✅ Break+")
            if r.get("ema_stack_full"):      flags.append("📐 EMA✓✓")
            if r.get("candle_ok"):           flags.append("🕯 Candle✓")
            if r.get("rs_ok"):               flags.append(f"💪 RS{r['rs_val']:+.1f}%")
            if r.get("atr_ok"):              flags.append("📡 ATR✓")
            if r.get("squeeze"):             flags.append("🗜 Squeeze")
        if r["vol_ratio"] >= 1.5:        flags.append(f"📈 Vol{r['vol_ratio']}x")
        if r["iv_rank"] is not None and r["iv_rank"] <= 30: flags.append("💰 LowIV")
        if r.get("earnings_near"):       flags.append(f"⚠️ Earn{r['days_to_earn']}d")

        row = {
            "Ticker":     r["symbol"],
            "Price":      f"${r['spot']:.2f}",
            "Chg %":      f"{r['chg']:+.2f}%",
            "Signal":     sig,
            "Score":      f"{r['score']}/{r.get('score_max',13)}",
            "Strength":   r["strength"],
            "ADX":        r["adx"],
            "RSI":        r["rsi_val"],
            "RS vs SPY":  f"{r['rs_val']:+.1f}%" if r.get("rs_val") is not None else "—",
            "Confluence": "  ".join(flags) if flags else "—",
        }
        if opt:
            row.update({
                "Strike":  f"${opt['strike']:.1f}",
                "Expiry":  f"{opt['expiry']} ({opt['dte']}d)",
                "IV":      f"{opt['iv']}%",
                "IV Rank": f"{r['iv_rank']:.0f}%" if r["iv_rank"] is not None else "—",
                "Spread":  f"{opt['spread_pct']:.1f}%",
                "Mid":     f"${opt['mid']:.2f}",
                "Vol/OI":  f"{opt['volume']:,} / {opt['oi']:,}",
            })
        else:
            row.update({"Strike":"—","Expiry":"—","IV":"—","IV Rank":"—","Spread":"—","Mid":"—","Vol/OI":"—"})
        table.append(row)

    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"📋 All ({len(rows)})",
    f"🟢 Calls ({len(calls)})",
    f"🔴 Puts ({len(puts)})",
    "🔍 Trade Builder",
    "🧪 Backtest",
])

with tab1: render_table(rows)
with tab2: render_table(calls)
with tab3: render_table(puts)

# ─── TRADE BUILDER ───────────────────────────────────────────────────────────────

with tab4:
    st.subheader("🔍 Trade Builder")
    actionable = [r for r in rows if r["opt"] is not None and r["signal"] != "WAIT"]
    if not actionable:
        st.info("No actionable signals with options data right now.")
    else:
        chosen = st.selectbox("Select ticker", [r["symbol"] for r in actionable])
        r = next((x for x in actionable if x["symbol"] == chosen), None)

        if r:
            opt = r["opt"]

            if r["earnings_near"]:
                st.warning(f"⚠️ **Earnings in {r['days_to_earn']} days ({r['earnings_date']})** — IV may crush after. Consider waiting or sizing down.")
            elif r["earnings_date"]:
                st.success(f"✅ Next earnings: {r['earnings_date']} ({r['days_to_earn']} days away) — safe window.")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Price",   f"${r['spot']:.2f}", f"{r['chg']:+.2f}%",
                      delta_color="normal" if r["chg"] >= 0 else "inverse")
            c2.metric("Signal",  r["signal"])
            c3.metric("Score",   f"{r['score']}/{r.get('score_max',13)}")
            c4.metric("ADX",     r["adx"])
            c5.metric("RSI",     r["rsi_val"])
            c6.metric("RS/SPY",  f"{r['rs_val']:+.1f}%")

            # ── Checklist ──────────────────────────────────────────────────────
            if r.get("strategy") == "mean_reversion":
                st.subheader("✅ Mean-Reversion Checklist")
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown("**🔄 Reversion Setup**")
                    rsi2v = r.get("rsi2", 50)
                    st.markdown(f"{'✅' if rsi2v < 10 or rsi2v > 90 else '🟡'} **RSI(2) = {rsi2v}** — {'extreme — snap-back likely ✓' if (rsi2v<10 or rsi2v>90) else 'not yet extreme'}")
                    st.markdown(f"{'✅' if r.get('adx_ok') else '❌'} **ADX {r['adx']}** — {'choppy regime ✓ (MR works here)' if r.get('adx_ok') else 'trending — MR riskier'}")
                    st.markdown(f"{'✅' if r['vol_ratio'] >= 1.5 else '⚪'} **Volume {r['vol_ratio']}x** — {'capitulation spike ✓' if r['vol_ratio']>=1.5 else 'normal'}")
                with mc2:
                    st.markdown("**💸 Tradeability**")
                    iv_cheap = r['iv_rank'] is not None and r['iv_rank'] <= 45
                    st.markdown(f"{'✅' if iv_cheap else '❌'} **IV Rank {r['iv_rank'] if r['iv_rank'] is not None else '?'}%** — {'cheap ✓' if iv_cheap else 'expensive'}")
                    spread_label = f"{opt['spread_pct']:.1f}%" if opt else "—"
                    st.markdown(f"{'✅' if r.get('spread_ok') else '❌'} **Spread {spread_label}** — {'fillable ✓' if r.get('spread_ok') else 'wide'}")
                    earn_txt = "safe ✓" if not r['earnings_near'] else f"{r['days_to_earn']}d — IV crush risk"
                    st.markdown(f"{'✅' if not r['earnings_near'] else '⚠️'} **Earnings** — {earn_txt}")
                st.warning("⚠️ Mean-reversion = catching a falling knife. Exit fast at the mean (SMA10), honor the −5% stop. Verify it's a *dip*, not a *collapse*.")
            elif r.get("strategy") == "chartprime_reversal":
                st.subheader("✅ Supertrend Reversal — Confluence Stack")
                cf = r.get("conf_flags", {})
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("**🎯 Confluences (need ≥2)**")
                    for lab, ok in cf.items():
                        st.markdown(f"{'✅' if ok else '❌'} **{lab}**")
                with rc2:
                    st.markdown("**💸 Tradeability**")
                    iv_cheap = r['iv_rank'] is not None and r['iv_rank'] <= 45
                    st.markdown(f"{'✅' if iv_cheap else '❌'} **IV Rank {r['iv_rank'] if r['iv_rank'] is not None else '?'}%**")
                    spread_label = f"{opt['spread_pct']:.1f}%" if opt else "—"
                    st.markdown(f"{'✅' if r.get('spread_ok') else '❌'} **Spread {spread_label}**")
                    earn_txt = "safe ✓" if not r['earnings_near'] else f"{r['days_to_earn']}d — IV crush risk"
                    st.markdown(f"{'✅' if not r['earnings_near'] else '⚠️'} **Earnings** — {earn_txt}")
                st.markdown(f"**Confluence score: {r['score']}/4**  ·  recommended exit: hold for the trend, opposite Supertrend flip or −4% stop.")
                st.warning("⚠️ **Use 30 DTE** (shorter expiries get eaten by theta). Regime-dependent — profitable ~2 of 4 years in backtest. Paper-trade first, size small, expect losing stretches.")
            elif r.get("strategy") == "topdown":
                st.subheader("✅ Top-Down Level Reclaim")
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.markdown("**🏔 Setup**")
                    st.markdown(f"✅ **Key level reclaimed** — 50-bar {'support' if r['signal']=='CALL' else 'resistance'} at ${r.get('td_level','?')}")
                    st.markdown(f"✅ **Trend-aligned** — price {'above' if r['signal']=='CALL' else 'below'} 200-EMA")
                    st.markdown(f"{'✅' if r.get('td_vol_ok') else '❌'} **Volatility {r.get('hv','?')}%** — {'high enough for options ✓' if r.get('td_vol_ok') else 'too low — skip'}")
                    st.markdown(f"{'✅' if r['vol_ratio']>=1.5 else '⚪'} **Volume {r['vol_ratio']}x**")
                with tc2:
                    st.markdown("**🛡 Risk (2:1)**")
                    st.markdown(f"🛑 **Stop:** ${r.get('td_stop_ref','?')}  ({r.get('td_stop_pct','?')}% away)")
                    st.markdown(f"🎯 **Target:** ${r.get('td_target','?')}  (2× risk)")
                    iv_cheap = r['iv_rank'] is not None and r['iv_rank'] <= 45
                    st.markdown(f"{'✅' if iv_cheap else '❌'} **IV Rank {r['iv_rank'] if r['iv_rank'] is not None else '?'}%**")
                    earn_txt = "safe ✓" if not r['earnings_near'] else f"{r['days_to_earn']}d — IV crush risk"
                    st.markdown(f"{'✅' if not r['earnings_near'] else '⚠️'} **Earnings** — {earn_txt}")
                st.warning("⚠️ **Use 30 DTE**, high-vol names only. Most robust setup in testing (survived out-of-sample + across universes) — but recent year was negative. Paper-trade first, honor the 2:1 stop.")
            else:
                st.subheader("✅ Anti-Fakeout Checklist (13 Factors)")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📊 Trend & Momentum**")
                    pts = r.get("breakout_pts", 0)
                    conf = "confirmed +0.5% clearance ✓" if r.get("confirmed") else "touched level only" if pts == 1 else "no breakout"
                    st.markdown(f"{'✅' if pts == 2 else '🟡' if pts == 1 else '❌'} **Donchian Breakout** ({pts}/2 pts) — {conf}")
                    eps = r.get("ema_stack_pts", 0)
                    st.markdown(f"{'✅' if r.get('ema_stack_full') else '🟡' if eps == 1 else '❌'} **EMA Stack 10>50>200** ({eps}/2 pts) — {'fully aligned ✓' if r.get('ema_stack_full') else 'partial' if eps == 1 else 'misaligned'}")
                    st.markdown(f"{'✅' if r.get('adx_ok') else '❌'} **ADX {r['adx']}** — trend {'confirmed' if r.get('adx_ok') else 'too weak'}")
                    st.markdown(f"{'✅' if r.get('macd_ok') else '❌'} **MACD** — {r.get('macd_dir','').replace('_',' ').title()}")
                    st.markdown(f"{'✅' if r['vol_ratio'] >= 1.5 else '❌'} **Volume Spike** — {r['vol_ratio']}x average")
                    st.markdown(f"{'✅' if r.get('squeeze') else '❌'} **BB Squeeze** — {'coil detected' if r.get('squeeze') else 'no squeeze'}")
                with col2:
                    st.markdown("**🛡️ Anti-Fakeout Filters**")
                    st.markdown(f"{'✅' if r.get('candle_ok') else '❌'} **Candle Quality** — {'strong close, no rejection wick ✓' if r.get('candle_ok') else 'weak/doji candle — fakeout risk'}")
                    st.markdown(f"{'✅' if r.get('rs_ok') else '❌'} **Relative Strength vs SPY** — {r.get('rs_val', 0):+.1f}% {'outperforming ✓' if r.get('rs_ok') and r['signal']=='CALL' else 'underperforming ✓' if r.get('rs_ok') and r['signal']=='PUT' else 'not confirming move'}")
                    st.markdown(f"{'✅' if r.get('atr_ok') else '❌'} **ATR Regime** — {r.get('atr_ratio', 0):.1f}x avg {'— normal volatility ✓' if r.get('atr_ok') else '— SPIKE detected, possible news gap'}")
                    st.markdown(f"{'✅' if r.get('spy_aligned') else '❌'} **SPY Regime {spy_regime}** — {'aligned ✓' if r.get('spy_aligned') else 'fighting the trend'}")
                    st.markdown(f"{'✅' if r.get('rsi_ok') else '❌'} **RSI {r['rsi_val']}** — {'momentum confirmed ✓' if r.get('rsi_ok') else 'momentum against direction'}")
                    iv_cheap = r['iv_rank'] is not None and r['iv_rank'] <= 45
                    st.markdown(f"{'✅' if iv_cheap else '❌'} **IV Rank {r['iv_rank'] if r['iv_rank'] is not None else '?'}%** — {'cheap premium ✓' if iv_cheap else 'expensive — consider waiting'}")
                    spread_label = f"{opt['spread_pct']:.1f}%" if opt else "—"
                    st.markdown(f"{'✅' if r.get('spread_ok') else '❌'} **Bid/Ask Spread {spread_label}** — {'fillable ✓' if r.get('spread_ok') else 'wide — hard to fill at mid'}")
                    earn_txt = "safe window ✓" if not r['earnings_near'] else f"{r['days_to_earn']}d away — IV crush risk"
                    st.markdown(f"{'✅' if not r['earnings_near'] else '⚠️'} **Earnings** — {earn_txt}")

            st.divider()

            # Contract
            st.subheader(f"📝 Recommended Contract — {opt['type']}")
            o1, o2, o3, o4, o5, o6, o7 = st.columns(7)
            o1.metric("Strike",  f"${opt['strike']:.1f}")
            o2.metric("Expiry",  opt["expiry"])
            o3.metric("DTE",     f"{opt['dte']}d")
            o4.metric("IV",      f"{opt['iv']}%")
            o5.metric("IV Rank", f"{r['iv_rank']:.0f}%" if r["iv_rank"] is not None else "—")
            o6.metric("Delta",   opt["delta"])
            o7.metric("Spread",  f"{opt['spread_pct']:.1f}%")

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Bid",    f"${opt['bid']:.2f}")
            p2.metric("Ask",    f"${opt['ask']:.2f}")
            p3.metric("Mid",    f"${opt['mid']:.2f}")
            p4.metric("Vol/OI", f"{opt['volume']:,} / {opt['oi']:,}")

            st.divider()

            # Position sizing
            st.subheader("💰 Position Sizing & Risk/Reward")
            account_size = st.number_input("Account size ($)", value=10000, step=1000)
            risk_pct     = st.slider("Risk per trade (%)", 1, 10, 3)

            mid       = opt["mid"]
            risk_amt  = account_size * risk_pct / 100
            contracts = max(1, int(risk_amt / (mid * 100 * 0.5))) if mid > 0 else 1
            cost      = round(contracts * mid * 100, 2)
            stop_px   = round(mid * 0.50, 2)
            tgt1_px   = round(mid * 1.50, 2)
            tgt2_px   = round(mid * 2.50, 2)
            max_loss  = round((mid - stop_px) * contracts * 100, 2)
            gain_tgt1 = round((tgt1_px - mid) * contracts * 100, 2)
            gain_tgt2 = round((tgt2_px - mid) * contracts * 100, 2)
            rr1 = round(gain_tgt1 / max_loss, 1) if max_loss > 0 else 0
            rr2 = round(gain_tgt2 / max_loss, 1) if max_loss > 0 else 0

            st.markdown(f"""
| | |
|---|---|
| **Entry (mid)** | ${mid:.2f} per contract |
| **Contracts** | {contracts} ({risk_pct}% risk = ~${cost:,.0f} total) |
| **Stop loss (−50%)** | ${stop_px:.2f} → max loss **−${max_loss:,.0f}** |
| **Target 1 (+50%)** | ${tgt1_px:.2f} → profit **+${gain_tgt1:,.0f}** (R/R 1:{rr1}) |
| **Target 2 (+150%)** | ${tgt2_px:.2f} → profit **+${gain_tgt2:,.0f}** (R/R 1:{rr2}) |
""")

            st.success(
                f"**Trade idea:** Buy {contracts}× ${opt['strike']} {opt['type']} "
                f"exp {opt['expiry']} @ ~${mid:.2f}. "
                f"Stop at 50% loss (${stop_px:.2f}). "
                f"Take half at +50% (${tgt1_px:.2f}), rest at +150% (${tgt2_px:.2f})."
            )

            # Chart
            st.divider()
            st.subheader(f"📈 {chosen} — 6-Month Price + EMA Stack")
            hist2 = fetch_bars(chosen, tf_label, open_wait_mins)
            if not hist2.empty:
                hist2["EMA10"]  = hist2["Close"].ewm(span=10,  adjust=False).mean()
                hist2["EMA50"]  = hist2["Close"].ewm(span=50,  adjust=False).mean()
                hist2["EMA200"] = hist2["Close"].ewm(span=200, adjust=False).mean()
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist2.index, y=hist2["Close"],  name="Close",  line=dict(color="#4CAF50", width=2)))
                fig.add_trace(go.Scatter(x=hist2.index, y=hist2["EMA10"],  name="EMA10",  line=dict(color="#2196F3", width=1.5, dash="dot")))
                fig.add_trace(go.Scatter(x=hist2.index, y=hist2["EMA50"],  name="EMA50",  line=dict(color="#FF9800", width=1.5, dash="dash")))
                fig.add_trace(go.Scatter(x=hist2.index, y=hist2["EMA200"], name="EMA200", line=dict(color="#E91E63", width=1.5, dash="longdash")))
                fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=340, legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

# ─── BACKTEST TAB ────────────────────────────────────────────────────────────────

with tab5:
    st.subheader("🧪 Vet a Ticker — Which Strategy Fits?")
    st.caption(
        "Walk-forward backtest on the **underlying** (Daily bars). **Trend** buys breakouts; "
        "**Mean-Reversion** fades extremes. Use *Compare Both* to see which one historically "
        "worked on a given name — they tend to be opposites (trenders vs choppy names)."
    )

    bc1, bc2, bc3 = st.columns([2,1,1])
    bt_symbol = bc1.text_input("Ticker", value="AMD", key="bt_sym").strip().upper()
    bt_window = bc2.selectbox("Lookback", ["3 months","6 months","1 year","2 years"], index=2)
    bt_mode   = bc3.selectbox("Strategy", ["Compare All","Trend only","Mean-Reversion only","Top-Down Levels only"], index=0)
    win_map   = {"3 months":63, "6 months":126, "1 year":252, "2 years":500}

    def _verdict(arr):
        wr = (arr > 0).mean()*100; exp = float(arr.mean())
        if exp > 0.3 and wr >= 45:  return "🟢 TRADEABLE", "success", wr, exp
        if exp > 0:                 return "🟡 MARGINAL", "warning", wr, exp
        return "🔴 AVOID", "error", wr, exp

    def _show_one(label, strat, score_floor):
        trades, table = run_backtest(bt_symbol, win_map[bt_window], score_floor, min_adx, strategy=strat)
        if trades is None:
            st.error(f"Couldn't fetch enough history for {bt_symbol}."); return None
        if not trades:
            st.warning(f"**{label}:** no signals at score ≥{score_floor} — strategy stayed out (not necessarily bad)."); return None
        arr = np.array(trades)
        verdict, vcolor, wr, exp = _verdict(arr)
        getattr(st, vcolor)(f"**{label} → {bt_symbol}: {verdict}**")
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Signals", len(arr)); k2.metric("Win Rate", f"{wr:.0f}%")
        k3.metric("Expectancy", f"{exp:+.2f}%"); k4.metric("Total", f"{arr.sum():+.1f}%")
        k5.metric("Best/Worst", f"{arr.max():+.0f}/{arr.min():+.0f}%")
        with st.expander(f"{label} — every trade"):
            st.dataframe(pd.DataFrame(table[::-1]), use_container_width=True, hide_index=True)
        return exp

    if st.button("▶️ Run Backtest", use_container_width=True):
        with st.spinner(f"Backtesting {bt_symbol}…"):
            if bt_mode == "Trend only":
                _show_one("📈 Trend", "trend", 9)
            elif bt_mode == "Mean-Reversion only":
                _show_one("🔄 Mean-Reversion", "mean_reversion", 6)
            elif bt_mode == "Top-Down Levels only":
                _show_one("🏔 Top-Down Levels", "topdown", 0)
            else:
                e_tr = _show_one("📈 Trend", "trend", 9)
                st.divider()
                e_mr = _show_one("🔄 Mean-Reversion", "mean_reversion", 6)
                st.divider()
                e_td = _show_one("🏔 Top-Down Levels", "topdown", 0)
                # Recommendation — pick the best of the three
                cands = [("📈 Trend", e_tr), ("🔄 Mean-Reversion", e_mr), ("🏔 Top-Down Levels", e_td)]
                cands = [(n, e) for n, e in cands if e is not None]
                if cands:
                    best_n, best_e = max(cands, key=lambda x: x[1])
                    if best_e <= 0:
                        st.info(f"🛑 **Recommendation:** No strategy has an edge on {bt_symbol} — **skip this name.**")
                    else:
                        st.success(f"✅ **Recommendation:** Trade **{bt_symbol}** with **{best_n}** (best expectancy: {best_e:+.2f}%/trade).")

        st.caption(
            "⚠️ Underlying returns only — real options add theta & spread. "
            "Trend: hold ≤10d, −4% stop, +4% trailing. Mean-Reversion: hold ≤7d, exit at mean (SMA10), −5% stop. "
            "Top-Down: 50-bar level reclaim + 200-EMA trend + HV≥40% gate, 2:1 risk."
        )

# ─── FOOTER ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ For educational & paper trading use only. Not financial advice. "
    "Options trading involves significant risk of loss."
)

if auto_refresh:
    time.sleep(300)
    st.cache_data.clear()
    st.rerun()
