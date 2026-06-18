#!/usr/bin/env python3
"""
Professional Options Scanner — Breakout + Cheap Premium Strategy
by Claude, for paper/live options trading

Entry logic:
  1. Donchian breakout (20-day high/low)        → direction + 2pts
  2. ADX > 20 (real trend, not chop)            → +1pt
  3. Bollinger Band squeeze (volatility coil)   → +1pt
  4. MACD histogram crossover                   → +1pt
  5. Volume spike > 1.5x avg                    → +1pt
  6. IV Rank < 45% (cheap premium)              → +1pt
  Score 4+ = Strong  |  3 = Medium  |  2 = Weak
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

# ─── WATCHLIST ───────────────────────────────────────────────────────────────────
WATCHLIST = [
    "SPY","QQQ","AAPL","TSLA","NVDA",
    "AMD","MSFT","AMZN","META","GOOGL",
    "NFLX","COIN","MSTR","PLTR","ARM",
    "SMCI","AVGO","MU","SOFI","HOOD",
]
MIN_OI = 50

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
.signal-strong { color:#00e676; font-weight:700; }
.signal-medium { color:#ffeb3b; font-weight:700; }
.signal-weak   { color:#ff7043; font-weight:700; }
.tag-call { background:#1b5e20; color:#a5d6a7; padding:2px 8px; border-radius:4px; font-weight:700; }
.tag-put  { background:#b71c1c; color:#ef9a9a; padding:2px 8px; border-radius:4px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Options Scanner Pro — Breakout + Cheap Premium")
st.caption(f"Strategy: Donchian breakout · ADX trend · BB squeeze · MACD · Volume · IV Rank  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filters")

    custom_input = st.text_input("➕ Add tickers (comma-separated)", placeholder="UBER, ROKU, SNAP")
    watchlist = WATCHLIST.copy()
    if custom_input:
        watchlist += [t.strip().upper() for t in custom_input.split(",") if t.strip()]
        watchlist = list(dict.fromkeys(watchlist))

    target_dte    = st.select_slider("Target DTE", options=[7,14,21,30,45,60], value=30)
    min_score     = st.slider("Min signal score (0–7)", 0, 7, 2)
    max_iv_rank   = st.slider("Max IV Rank % (cheaper = lower)", 0, 100, 50)
    min_adx       = st.slider("Min ADX (trend strength)", 10, 40, 18)

    st.divider()
    show_wait = st.toggle("Show 'WAIT' signals too", value=False)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    auto_refresh = st.toggle("Auto-refresh every 5 min", value=False)

    st.divider()
    st.markdown("""
**Strategy Logic**
- 🟢 **CALL** — Bullish breakout, cheap IV
- 🔴 **PUT** — Bearish breakdown, cheap IV
- ⚪ **WAIT** — No edge / IV too expensive

**Score**
- 🔥 5–7 pts → Strong
- ⚡ 3–4 pts → Medium
- 💤 1–2 pts → Weak

**Why IV Rank matters**
Buying options when IV is low means
you pay less for the same contract.
When IV expands you profit on *vega*
in addition to directional move.
    """)


# ─── INDICATORS ──────────────────────────────────────────────────────────────────

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()


def donchian_signal(close: pd.Series, period=20):
    """Break above 20-day high = bullish, below 20-day low = bearish."""
    high20 = close.rolling(period).max().shift(1)
    low20  = close.rolling(period).min().shift(1)
    last   = close.iloc[-1]
    if last >= high20.iloc[-1]:
        return "BULLISH", 2
    elif last <= low20.iloc[-1]:
        return "BEARISH", 2
    return "NEUTRAL", 0


def adx(high, low, close, period=14):
    """Average Directional Index — measures trend strength."""
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

    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean()  / atr
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr
    dx       = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan))
    adx_val  = dx.ewm(span=period, adjust=False).mean()
    return float(adx_val.iloc[-1]), float(di_plus.iloc[-1]), float(di_minus.iloc[-1])


def bb_squeeze(close: pd.Series, period=20):
    """Bollinger Band squeeze — narrow bands = coiling for a move."""
    mid  = sma(close, period)
    std  = close.rolling(period).std()
    bw   = (std * 4) / mid             # bandwidth
    bw_min = bw.rolling(125).min()    # 6-month low bandwidth
    squeezed = bw.iloc[-1] <= bw_min.iloc[-1] * 1.05
    return squeezed


def macd_signal(close: pd.Series):
    """MACD histogram crossing zero."""
    macd_line = ema(close, 12) - ema(close, 26)
    signal    = ema(macd_line, 9)
    hist      = macd_line - signal
    cross_up  = hist.iloc[-1] > 0 and hist.iloc[-2] <= 0
    cross_dn  = hist.iloc[-1] < 0 and hist.iloc[-2] >= 0
    if cross_up:
        return "BULLISH"
    elif cross_dn:
        return "BEARISH"
    # Histogram trending
    if hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3]:
        return "BULLISH_TREND"
    elif hist.iloc[-1] < hist.iloc[-2] < hist.iloc[-3]:
        return "BEARISH_TREND"
    return "NEUTRAL"


def rsi(close: pd.Series, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def historical_vol(close: pd.Series, period=20):
    returns = np.log(close / close.shift()).dropna()
    return float(returns.rolling(period).std().iloc[-1] * np.sqrt(252) * 100)


def iv_rank_estimate(ticker_obj, current_iv):
    """Estimate IV rank: where does current IV sit in its 1-year range?"""
    try:
        hist = ticker_obj.history(period="1y")
        if len(hist) < 60:
            return None
        returns  = np.log(hist["Close"] / hist["Close"].shift()).dropna()
        hv_series = returns.rolling(20).std() * np.sqrt(252) * 100
        hv_min   = hv_series.quantile(0.05)
        hv_max   = hv_series.quantile(0.95)
        if hv_max <= hv_min:
            return None
        rank = max(0, min(100, (current_iv - hv_min) / (hv_max - hv_min) * 100))
        return round(rank, 1)
    except Exception:
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
        if direction == "BULLISH":
            pool = df[df["strike"] >= spot * 0.99]
        else:
            pool = df[df["strike"] <= spot * 1.01]
        row = (pool if not pool.empty else df).sort_values("dist").iloc[0]

        iv  = round(float(row.get("impliedVolatility", 0) or 0) * 100, 1)
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        mid = round((bid + ask) / 2, 2) if bid and ask else round(float(row.get("lastPrice", 0) or 0), 2)

        return {
            "type":   "CALL" if direction == "BULLISH" else "PUT",
            "expiry": best_exp,
            "dte":    dte,
            "strike": float(row["strike"]),
            "iv":     iv,
            "delta":  round(abs(float(row.get("delta", 0.5) or 0.5)), 2),
            "volume": int(row.get("volume", 0) or 0),
            "oi":     int(row.get("openInterest", 0) or 0),
            "bid":    bid,
            "ask":    ask,
            "mid":    mid,
        }
    except Exception:
        return None


# ─── CORE SCANNER ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def scan(symbol, target_dte, min_adx_val):
    try:
        tk   = yf.Ticker(symbol)
        hist = tk.history(period="1y", interval="1d")
        if len(hist) < 60:
            return None

        close  = hist["Close"]
        high   = hist["High"]
        low    = hist["Low"]
        volume = hist["Volume"]
        spot   = float(close.iloc[-1])
        prev   = float(close.iloc[-2])
        chg    = round((spot - prev) / prev * 100, 2)

        # ── Factor 1: Donchian breakout (2 pts) ──────────────────────────────
        direction, don_score = donchian_signal(close, 20)
        score = don_score

        # ── Factor 2: ADX trend strength (1 pt) ──────────────────────────────
        adx_val, di_plus, di_minus = adx(high, low, close)
        adx_val = round(adx_val, 1)
        trending = adx_val >= min_adx_val
        if trending:
            # Also confirm ADX direction aligns
            if direction == "BULLISH" and di_plus > di_minus:
                score += 1
            elif direction == "BEARISH" and di_minus > di_plus:
                score += 1

        # ── Factor 3: BB squeeze (1 pt) ──────────────────────────────────────
        squeeze = bb_squeeze(close)
        if squeeze:
            score += 1

        # ── Factor 4: MACD (1 pt) ────────────────────────────────────────────
        macd_dir = macd_signal(close)
        macd_ok  = (
            (direction == "BULLISH" and macd_dir in ("BULLISH", "BULLISH_TREND")) or
            (direction == "BEARISH" and macd_dir in ("BEARISH", "BEARISH_TREND"))
        )
        if macd_ok:
            score += 1

        # ── Factor 5: Volume spike (1 pt) ────────────────────────────────────
        vol_avg   = float(volume.iloc[-21:-1].mean())
        vol_today = float(volume.iloc[-1])
        vol_ratio = round(vol_today / vol_avg, 2) if vol_avg > 0 else 1.0
        if vol_ratio >= 1.5:
            score += 1

        # ── Factor 6: IV rank < max_iv_rank (1 pt — checked after options fetch) ──
        rsi_val  = round(rsi(close), 1)
        hv_val   = round(historical_vol(close), 1)

        # Determine signal
        if direction == "NEUTRAL" or score < 2:
            signal = "WAIT"
        elif direction == "BULLISH":
            signal = "CALL"
        else:
            signal = "PUT"

        # Strength label
        if score >= 5:
            strength, strength_val = "🔥 Strong", 3
        elif score >= 3:
            strength, strength_val = "⚡ Medium", 2
        else:
            strength, strength_val = "💤 Weak",   1

        # Fetch options
        opt      = None
        iv_rank  = None
        if signal != "WAIT":
            opt = best_contract(tk, direction, spot, target_dte)
            if opt:
                iv_rank = iv_rank_estimate(tk, opt["iv"])
                if iv_rank is not None and iv_rank <= 45:
                    score  += 1
                    if score >= 5:
                        strength, strength_val = "🔥 Strong", 3
                    elif score >= 3:
                        strength, strength_val = "⚡ Medium", 2

        return {
            "symbol":       symbol,
            "spot":         spot,
            "chg":          chg,
            "signal":       signal,
            "direction":    direction,
            "score":        score,
            "strength":     strength,
            "strength_val": strength_val,
            "adx":          adx_val,
            "squeeze":      squeeze,
            "macd":         macd_dir,
            "rsi":          rsi_val,
            "hv":           hv_val,
            "vol_ratio":    vol_ratio,
            "opt":          opt,
            "iv_rank":      iv_rank,
        }
    except Exception:
        return None


# ─── SCAN ────────────────────────────────────────────────────────────────────────

bar  = st.progress(0, text="Scanning…")
rows = []
for i, sym in enumerate(watchlist):
    bar.progress((i + 1) / len(watchlist), text=f"Scanning {sym}…")
    r = scan(sym, target_dte, min_adx)
    if r and r["score"] >= min_score:
        if show_wait or r["signal"] != "WAIT":
            if r["opt"] is None or (r["iv_rank"] is None or r["iv_rank"] <= max_iv_rank):
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

        # Confluence flags
        flags = []
        if r["squeeze"]:           flags.append("🗜 Squeeze")
        if r["vol_ratio"] >= 1.5:  flags.append(f"📈 Vol {r['vol_ratio']}x")
        if r["adx"] >= 25:         flags.append(f"↗ ADX {r['adx']}")
        if "BULLISH" in r["macd"]: flags.append("✅ MACD")
        if "BEARISH" in r["macd"]: flags.append("✅ MACD")
        if r["iv_rank"] is not None and r["iv_rank"] <= 30:
            flags.append("💰 Low IV")

        row = {
            "Ticker":     r["symbol"],
            "Price":      f"${r['spot']:.2f}",
            "Chg %":      f"{r['chg']:+.2f}%",
            "Signal":     sig,
            "Score":      f"{r['score']}/7",
            "Strength":   r["strength"],
            "ADX":        r["adx"],
            "RSI":        r["rsi"],
            "Confluence": "  ".join(flags) if flags else "—",
        }
        if opt:
            row.update({
                "Strike":   f"${opt['strike']:.1f}",
                "Expiry":   f"{opt['expiry']} ({opt['dte']}d)",
                "IV":       f"{opt['iv']}%",
                "IV Rank":  f"{r['iv_rank']:.0f}%" if r["iv_rank"] is not None else "—",
                "Mid":      f"${opt['mid']:.2f}",
                "Vol/OI":   f"{opt['volume']:,} / {opt['oi']:,}",
            })
        else:
            row.update({"Strike": "—", "Expiry": "—", "IV": "—",
                        "IV Rank": "—", "Mid": "—", "Vol/OI": "—"})
        table.append(row)

    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)


tab1, tab2, tab3, tab4 = st.tabs([
    f"📋 All ({len(rows)})",
    f"🟢 Calls ({len(calls)})",
    f"🔴 Puts ({len(puts)})",
    "🔍 Trade Builder",
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

            # Header metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price",    f"${r['spot']:.2f}",   f"{r['chg']:+.2f}%",
                      delta_color="normal" if r["chg"] >= 0 else "inverse")
            c2.metric("Signal",   r["signal"])
            c3.metric("Score",    f"{r['score']}/7")
            c4.metric("ADX",      r["adx"])
            c5.metric("RSI",      r["rsi"])

            # Confluence checklist
            st.subheader("✅ Confluence Checklist")
            cl1, cl2 = st.columns(2)
            with cl1:
                st.markdown(f"{'✅' if r['direction'] != 'NEUTRAL' else '❌'} **Donchian Breakout** — price breaking 20-day {'high' if r['signal']=='CALL' else 'low'}")
                st.markdown(f"{'✅' if r['adx'] >= min_adx else '❌'} **ADX {r['adx']}** — trend strength {'confirmed' if r['adx'] >= min_adx else 'weak'}")
                st.markdown(f"{'✅' if r['squeeze'] else '❌'} **BB Squeeze** — {'volatility coil detected' if r['squeeze'] else 'no squeeze'}")
            with cl2:
                st.markdown(f"{'✅' if 'BULLISH' in r['macd'] or 'BEARISH' in r['macd'] else '❌'} **MACD** — {r['macd'].replace('_',' ').title()}")
                st.markdown(f"{'✅' if r['vol_ratio'] >= 1.5 else '❌'} **Volume Spike** — {r['vol_ratio']}x average")
                iv_cheap = r['iv_rank'] is not None and r['iv_rank'] <= 45
                st.markdown(f"{'✅' if iv_cheap else '❌'} **IV Rank {r['iv_rank'] if r['iv_rank'] is not None else '?'}%** — {'cheap premium ✓' if iv_cheap else 'premium expensive'}")

            st.divider()

            # Contract details
            st.subheader(f"📝 Recommended Contract — {opt['type']}")
            o1, o2, o3, o4, o5, o6 = st.columns(6)
            o1.metric("Strike",  f"${opt['strike']:.1f}")
            o2.metric("Expiry",  opt["expiry"])
            o3.metric("DTE",     f"{opt['dte']}d")
            o4.metric("IV",      f"{opt['iv']}%")
            o5.metric("IV Rank", f"{r['iv_rank']:.0f}%" if r["iv_rank"] is not None else "—")
            o6.metric("Delta",   opt["delta"])

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Bid",    f"${opt['bid']:.2f}")
            p2.metric("Ask",    f"${opt['ask']:.2f}")
            p3.metric("Mid",    f"${opt['mid']:.2f}")
            p4.metric("Volume", f"{opt['volume']:,}")

            st.divider()

            # Position sizing + R/R
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

            entry_type = "calls" if opt["type"] == "CALL" else "puts"
            st.success(
                f"**Trade idea:** Buy {contracts}× ${opt['strike']} {opt['type']} "
                f"exp {opt['expiry']} @ ~${mid:.2f}. "
                f"Stop at 50% loss (${stop_px:.2f}). "
                f"Take half off at +50% (${tgt1_px:.2f}), rest at +150% (${tgt2_px:.2f})."
            )

            # Price chart
            st.divider()
            st.subheader(f"📈 {chosen} — 6-Month Price")
            tk   = yf.Ticker(chosen)
            hist = tk.history(period="6mo")
            if not hist.empty:
                # Add EMAs to chart
                hist["EMA10"] = hist["Close"].ewm(span=10, adjust=False).mean()
                hist["EMA40"] = hist["Close"].ewm(span=40, adjust=False).mean()
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Close", line=dict(color="#4CAF50", width=2)))
                fig.add_trace(go.Scatter(x=hist.index, y=hist["EMA10"], name="EMA10", line=dict(color="#2196F3", width=1.5, dash="dot")))
                fig.add_trace(go.Scatter(x=hist.index, y=hist["EMA40"], name="EMA40", line=dict(color="#FF9800", width=1.5, dash="dot")))
                fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=300, legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

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
