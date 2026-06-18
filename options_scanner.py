#!/usr/bin/env python3
"""
Professional Options Scanner — Maximum Confluence Strategy
by Claude

Scoring (0–10 pts):
  1. Donchian breakout (20-day high/low)        → direction + 2pts
  2. ADX > 20 aligned with direction             → +1pt
  3. Bollinger Band squeeze                      → +1pt
  4. MACD histogram aligned                      → +1pt
  5. Volume spike > 1.5x avg                    → +1pt
  6. IV Rank < 45% (cheap premium)              → +1pt
  7. SPY market regime aligned                  → +1pt  [NEW]
  8. RSI not extreme (not overbought/oversold)  → +1pt  [NEW]
  9. Tight bid/ask spread < 15% of mid          → +1pt  [NEW]
 10. No earnings within 5 days                  → +1pt  [NEW]

  Strong 7–10  |  Medium 5–6  |  Weak 3–4
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
.earnings-warn { background:#7b3f00; color:#ffcc80; padding:4px 10px; border-radius:4px; font-weight:700; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Options Scanner Pro — Maximum Confluence")
st.caption(f"10-factor strategy: Breakout · ADX · BB Squeeze · MACD · Volume · IV Rank · SPY Regime · RSI · Spread · Earnings  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filters")

    custom_input = st.text_input("➕ Add tickers (comma-separated)", placeholder="UBER, ROKU, SNAP")
    watchlist = WATCHLIST.copy()
    if custom_input:
        watchlist += [t.strip().upper() for t in custom_input.split(",") if t.strip()]
        watchlist = list(dict.fromkeys(watchlist))

    target_dte    = st.select_slider("Target DTE", options=[7,14,21,30,45,60], value=30)
    min_score     = st.slider("Min signal score (0–10)", 0, 10, 4)
    max_iv_rank   = st.slider("Max IV Rank % (cheaper = lower)", 0, 100, 50)
    min_adx       = st.slider("Min ADX (trend strength)", 10, 40, 18)
    max_spread_pct = st.slider("Max bid/ask spread %", 5, 50, 20)

    st.divider()
    skip_earnings = st.toggle("Skip stocks with earnings ≤5 days", value=True)
    show_wait     = st.toggle("Show 'WAIT' signals too", value=False)
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
- 🔥 7–10 pts → Strong
- ⚡ 5–6 pts → Medium
- 💤 3–4 pts → Weak

**New Filters**
- 📊 **SPY Regime** — only take calls in bull market, puts in bear
- 📉 **RSI Filter** — avoid buying overbought calls or oversold puts
- 💸 **Spread Filter** — skip contracts with wide spreads (hard to fill)
- 📅 **Earnings** — avoid IV crush from upcoming earnings
    """)


# ─── INDICATORS ──────────────────────────────────────────────────────────────────

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()


def donchian_signal(close: pd.Series, period=20):
    high20 = close.rolling(period).max().shift(1)
    low20  = close.rolling(period).min().shift(1)
    last   = close.iloc[-1]
    if last >= high20.iloc[-1]:
        return "BULLISH", 2
    elif last <= low20.iloc[-1]:
        return "BEARISH", 2
    return "NEUTRAL", 0


def adx(high, low, close, period=14):
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
    mid      = sma(close, period)
    std      = close.rolling(period).std()
    bw       = (std * 4) / mid
    bw_min   = bw.rolling(125).min()
    return bw.iloc[-1] <= bw_min.iloc[-1] * 1.05


def macd_signal(close: pd.Series):
    macd_line = ema(close, 12) - ema(close, 26)
    signal    = ema(macd_line, 9)
    hist      = macd_line - signal
    cross_up  = hist.iloc[-1] > 0 and hist.iloc[-2] <= 0
    cross_dn  = hist.iloc[-1] < 0 and hist.iloc[-2] >= 0
    if cross_up:   return "BULLISH"
    if cross_dn:   return "BEARISH"
    if hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3]: return "BULLISH_TREND"
    if hist.iloc[-1] < hist.iloc[-2] < hist.iloc[-3]: return "BEARISH_TREND"
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
    try:
        hist = ticker_obj.history(period="1y")
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


@st.cache_data(ttl=3600, show_spinner=False)
def get_spy_regime():
    """Returns 'BULL' if SPY is above its 50-day EMA, 'BEAR' otherwise."""
    try:
        spy  = yf.Ticker("SPY").history(period="6mo")
        c    = spy["Close"]
        above = float(c.iloc[-1]) > float(ema(c, 50).iloc[-1])
        return "BULL" if above else "BEAR"
    except Exception:
        return "UNKNOWN"


@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_date(symbol: str):
    """Returns next earnings date as a date object, or None."""
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


# ─── CORE SCANNER ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def scan(symbol, target_dte, min_adx_val, spy_regime):
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

        score = 0

        # ── 1. Donchian breakout (2 pts) ─────────────────────────────────────
        direction, don_score = donchian_signal(close, 20)
        score += don_score

        # ── 2. ADX trend (1 pt) ──────────────────────────────────────────────
        adx_val, di_plus, di_minus = adx(high, low, close)
        adx_val = round(adx_val, 1)
        if adx_val >= min_adx_val:
            if direction == "BULLISH" and di_plus > di_minus:
                score += 1
            elif direction == "BEARISH" and di_minus > di_plus:
                score += 1

        # ── 3. BB squeeze (1 pt) ─────────────────────────────────────────────
        squeeze = bb_squeeze(close)
        if squeeze:
            score += 1

        # ── 4. MACD (1 pt) ───────────────────────────────────────────────────
        macd_dir = macd_signal(close)
        macd_ok  = (
            (direction == "BULLISH" and macd_dir in ("BULLISH", "BULLISH_TREND")) or
            (direction == "BEARISH" and macd_dir in ("BEARISH", "BEARISH_TREND"))
        )
        if macd_ok:
            score += 1

        # ── 5. Volume spike (1 pt) ───────────────────────────────────────────
        vol_avg   = float(volume.iloc[-21:-1].mean())
        vol_today = float(volume.iloc[-1])
        vol_ratio = round(vol_today / vol_avg, 2) if vol_avg > 0 else 1.0
        if vol_ratio >= 1.5:
            score += 1

        # ── 6. RSI not extreme (1 pt) [NEW] ──────────────────────────────────
        rsi_val = round(rsi(close), 1)
        rsi_ok  = False
        if direction == "BULLISH" and 40 <= rsi_val <= 75:
            rsi_ok = True
            score += 1
        elif direction == "BEARISH" and 25 <= rsi_val <= 60:
            rsi_ok = True
            score += 1
        elif direction == "NEUTRAL":
            rsi_ok = True  # neutral — don't penalise

        # ── 7. SPY market regime (1 pt) [NEW] ────────────────────────────────
        spy_aligned = False
        if symbol != "SPY":
            if direction == "BULLISH" and spy_regime == "BULL":
                spy_aligned = True
                score += 1
            elif direction == "BEARISH" and spy_regime == "BEAR":
                spy_aligned = True
                score += 1
        else:
            spy_aligned = True  # SPY always counts for itself

        # ── 8. Earnings proximity (1 pt) [NEW] ───────────────────────────────
        earnings_date  = get_earnings_date(symbol)
        today          = datetime.today().date()
        days_to_earn   = (earnings_date - today).days if earnings_date else None
        earnings_near  = days_to_earn is not None and 0 <= days_to_earn <= 5
        earnings_score = not earnings_near  # pass = no earnings this week
        if earnings_score:
            score += 1

        hv_val = round(historical_vol(close), 1)

        # Determine signal
        if direction == "NEUTRAL" or score < 3:
            signal = "WAIT"
        elif direction == "BULLISH":
            signal = "CALL"
        else:
            signal = "PUT"

        # Strength label
        if score >= 7:
            strength, strength_val = "🔥 Strong", 3
        elif score >= 5:
            strength, strength_val = "⚡ Medium", 2
        else:
            strength, strength_val = "💤 Weak",   1

        # ── 9+10. Options: IV rank (1 pt) + tight spread (1 pt) [NEW] ────────
        opt     = None
        iv_rank = None
        spread_ok = False

        if signal != "WAIT":
            opt = best_contract(tk, direction, spot, target_dte)
            if opt:
                iv_rank = iv_rank_estimate(tk, opt["iv"])
                if iv_rank is not None and iv_rank <= 45:
                    score += 1
                    if score >= 7:
                        strength, strength_val = "🔥 Strong", 3
                    elif score >= 5:
                        strength, strength_val = "⚡ Medium", 2

                # Tight spread check
                if opt["spread_pct"] < 15:
                    spread_ok = True
                    score += 1
                    if score >= 7:
                        strength, strength_val = "🔥 Strong", 3
                    elif score >= 5:
                        strength, strength_val = "⚡ Medium", 2

        return {
            "symbol":        symbol,
            "spot":          spot,
            "chg":           chg,
            "signal":        signal,
            "direction":     direction,
            "score":         score,
            "strength":      strength,
            "strength_val":  strength_val,
            "adx":           adx_val,
            "squeeze":       squeeze,
            "macd":          macd_dir,
            "rsi":           rsi_val,
            "rsi_ok":        rsi_ok,
            "hv":            hv_val,
            "vol_ratio":     vol_ratio,
            "spy_aligned":   spy_aligned,
            "spy_regime":    spy_regime,
            "earnings_date": str(earnings_date) if earnings_date else None,
            "days_to_earn":  days_to_earn,
            "earnings_near": earnings_near,
            "spread_ok":     spread_ok,
            "opt":           opt,
            "iv_rank":       iv_rank,
        }
    except Exception:
        return None


# ─── SCAN ────────────────────────────────────────────────────────────────────────

spy_regime = get_spy_regime()
spy_label  = "🐂 Bull Market (SPY above 50 EMA)" if spy_regime == "BULL" else "🐻 Bear Market (SPY below 50 EMA)" if spy_regime == "BEAR" else "❓ Unknown"
st.info(f"**Market Regime:** {spy_label} — scanner favors {'CALLS' if spy_regime == 'BULL' else 'PUTS' if spy_regime == 'BEAR' else 'both directions'}")

bar  = st.progress(0, text="Scanning…")
rows = []
for i, sym in enumerate(watchlist):
    bar.progress((i + 1) / len(watchlist), text=f"Scanning {sym}…")
    r = scan(sym, target_dte, min_adx, spy_regime)
    if r is None:
        continue
    if r["score"] < min_score:
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
        if r["squeeze"]:                      flags.append("🗜 Squeeze")
        if r["vol_ratio"] >= 1.5:             flags.append(f"📈 Vol {r['vol_ratio']}x")
        if r["adx"] >= 25:                    flags.append(f"↗ ADX {r['adx']}")
        if "BULLISH" in r["macd"] or "BEARISH" in r["macd"]: flags.append("✅ MACD")
        if r["iv_rank"] is not None and r["iv_rank"] <= 30:   flags.append("💰 Low IV")
        if r["spy_aligned"]:                  flags.append("📊 SPY✓")
        if r["rsi_ok"]:                       flags.append(f"RSI {r['rsi']}")
        if r.get("spread_ok"):                flags.append("💸 Tight")
        if r["earnings_near"]:                flags.append(f"⚠️ Earn {r['days_to_earn']}d")

        row = {
            "Ticker":     r["symbol"],
            "Price":      f"${r['spot']:.2f}",
            "Chg %":      f"{r['chg']:+.2f}%",
            "Signal":     sig,
            "Score":      f"{r['score']}/10",
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
                "Spread":   f"{opt['spread_pct']:.1f}%",
                "Mid":      f"${opt['mid']:.2f}",
                "Vol/OI":   f"{opt['volume']:,} / {opt['oi']:,}",
            })
        else:
            row.update({"Strike":"—","Expiry":"—","IV":"—","IV Rank":"—","Spread":"—","Mid":"—","Vol/OI":"—"})
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

            # Earnings warning banner
            if r["earnings_near"]:
                st.warning(f"⚠️ **Earnings in {r['days_to_earn']} days ({r['earnings_date']})** — IV may spike before and crush after. Consider waiting or sizing down.")
            elif r["earnings_date"]:
                st.success(f"✅ Next earnings: {r['earnings_date']} ({r['days_to_earn']} days away) — safe window.")

            # Header metrics
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Price",   f"${r['spot']:.2f}", f"{r['chg']:+.2f}%",
                      delta_color="normal" if r["chg"] >= 0 else "inverse")
            c2.metric("Signal",  r["signal"])
            c3.metric("Score",   f"{r['score']}/10")
            c4.metric("ADX",     r["adx"])
            c5.metric("RSI",     r["rsi"])
            c6.metric("Regime",  r["spy_regime"])

            # Confluence checklist
            st.subheader("✅ Confluence Checklist (10 Factors)")
            cl1, cl2 = st.columns(2)
            with cl1:
                st.markdown(f"{'✅' if r['direction'] != 'NEUTRAL' else '❌'} **Donchian Breakout** — price breaking 20-day {'high' if r['signal']=='CALL' else 'low'}")
                st.markdown(f"{'✅' if r['adx'] >= min_adx else '❌'} **ADX {r['adx']}** — trend {'confirmed' if r['adx'] >= min_adx else 'weak'}")
                st.markdown(f"{'✅' if r['squeeze'] else '❌'} **BB Squeeze** — {'coil detected' if r['squeeze'] else 'no squeeze'}")
                st.markdown(f"{'✅' if 'BULLISH' in r['macd'] or 'BEARISH' in r['macd'] else '❌'} **MACD** — {r['macd'].replace('_',' ').title()}")
                st.markdown(f"{'✅' if r['vol_ratio'] >= 1.5 else '❌'} **Volume Spike** — {r['vol_ratio']}x average")
            with cl2:
                iv_cheap = r['iv_rank'] is not None and r['iv_rank'] <= 45
                st.markdown(f"{'✅' if iv_cheap else '❌'} **IV Rank {r['iv_rank'] if r['iv_rank'] is not None else '?'}%** — {'cheap premium ✓' if iv_cheap else 'expensive'}")
                st.markdown(f"{'✅' if r['rsi_ok'] else '❌'} **RSI {r['rsi']}** — {'healthy zone' if r['rsi_ok'] else 'extreme — avoid'}")
                st.markdown(f"{'✅' if r['spy_aligned'] else '❌'} **SPY Regime {r['spy_regime']}** — {'aligned ✓' if r['spy_aligned'] else 'fighting the trend'}")
                st.markdown(f"{'✅' if not r['earnings_near'] else '⚠️'} **Earnings** — {'safe ✓' if not r['earnings_near'] else f\"in {r['days_to_earn']} days — risky\"}")
                spread_label = f"{opt['spread_pct']:.1f}%" if opt else "—"
                st.markdown(f"{'✅' if r.get('spread_ok') else '❌'} **Bid/Ask Spread {spread_label}** — {'tight, easy to fill ✓' if r.get('spread_ok') else 'wide — hard to fill'}")

            st.divider()

            # Contract details
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
                f"Take half off at +50% (${tgt1_px:.2f}), rest at +150% (${tgt2_px:.2f})."
            )

            # Price chart
            st.divider()
            st.subheader(f"📈 {chosen} — 6-Month Price + EMAs")
            tk2  = yf.Ticker(chosen)
            hist2 = tk2.history(period="6mo")
            if not hist2.empty:
                hist2["EMA10"] = hist2["Close"].ewm(span=10, adjust=False).mean()
                hist2["EMA40"] = hist2["Close"].ewm(span=40, adjust=False).mean()
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist2.index, y=hist2["Close"], name="Close", line=dict(color="#4CAF50", width=2)))
                fig.add_trace(go.Scatter(x=hist2.index, y=hist2["EMA10"], name="EMA10", line=dict(color="#2196F3", width=1.5, dash="dot")))
                fig.add_trace(go.Scatter(x=hist2.index, y=hist2["EMA40"], name="EMA40", line=dict(color="#FF9800", width=1.5, dash="dot")))
                fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=320, legend=dict(orientation="h"))
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
