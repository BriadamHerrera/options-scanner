#!/usr/bin/env python3
"""
Backtest the Anti-Fakeout scanner's directional signal on a single ticker.

Backtests the UNDERLYING directional signal (not the option itself — historical
options chains aren't freely available). When a CALL/PUT fires at a bar's close,
we measure the stock's forward return over the next `HOLD_BARS` bars and apply a
stop/target on the underlying. A real option would amplify these moves.

Usage:  python3 backtest.py TSLA 15
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ─── PARAMS ──────────────────────────────────────────────────────────────────────
SYMBOL    = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
EVAL_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 15  # trading days to evaluate
MIN_SCORE  = 9       # raised from 8 — scores 6-8 showed no edge
MIN_ADX    = 22
HOLD_BARS  = 10      # extended from 3 — longer holds tripled expectancy
STOP_PCT   = 4.0     # initial underlying stop %
TGT_PCT    = 12.0    # final target (let winners run; trailing stop does the real work)
TRAIL_PCT  = 4.0     # trailing stop: give back 4% from the peak favorable price
TRAIL_ARM  = 4.0     # only start trailing once trade is +4% in our favor

# ─── INDICATORS (mirror options_scanner.py) ──────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()

def confirmed_donchian(close, period=20, clearance=0.005):
    high20 = close.rolling(period).max().shift(1)
    low20  = close.rolling(period).min().shift(1)
    c = close.iloc[-1]
    if c > high20.iloc[-1] * (1 + clearance):  return "BULLISH", 2
    if c < low20.iloc[-1]  * (1 - clearance):  return "BEARISH", 2
    if c >= high20.iloc[-1]:                   return "BULLISH", 1
    if c <= low20.iloc[-1]:                    return "BEARISH", 1
    return "NEUTRAL", 0

def ema_stack(close):
    e10, e50, e200 = float(ema(close,10).iloc[-1]), float(ema(close,50).iloc[-1]), float(ema(close,200).iloc[-1])
    last = float(close.iloc[-1])
    if last > e10 > e50 > e200: return "BULLISH", 2
    if last < e10 < e50 < e200: return "BEARISH", 2
    if last > e50 and e10 > e50: return "BULLISH", 1
    if last < e50 and e10 < e50: return "BEARISH", 1
    return "NEUTRAL", 0

def candle_quality(c, o, h, l):
    c,o,h,l = float(c), float(o), float(h), float(l)
    rng = h - l
    if rng == 0: return False, False
    body_pct  = abs(c-o)/rng
    close_pct = (c-l)/rng
    return (body_pct>=0.4 and close_pct>=0.6), (body_pct>=0.4 and close_pct<=0.4)

def adx_calc(high, low, close, period=14):
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    dmp = (high-high.shift()).clip(lower=0); dmm = (low.shift()-low).clip(lower=0)
    dmp = dmp.where(dmp>dmm,0); dmm = dmm.where(dmm>dmp,0)
    dip = 100*dmp.ewm(span=period,adjust=False).mean()/atr
    dim = 100*dmm.ewm(span=period,adjust=False).mean()/atr
    dx  = 100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return float(dx.ewm(span=period,adjust=False).mean().iloc[-1]), float(dip.iloc[-1]), float(dim.iloc[-1])

def atr_regime(high, low, close, period=14):
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    ap = tr/close
    avg = float(ap.rolling(252).mean().iloc[-1])
    if avg == 0: return True, 1.0
    r = float(ap.iloc[-1])/avg
    return r < 1.5, round(r,2)

def bb_squeeze(close, period=20):
    mid = sma(close,period); std = close.rolling(period).std()
    bw = (std*4)/mid; bw_min = bw.rolling(125).min()
    return bool(bw.iloc[-1] <= bw_min.iloc[-1]*1.05)

def macd_dir(close):
    ml = ema(close,12)-ema(close,26); sig = ema(ml,9); h = ml-sig
    if h.iloc[-1]>0 and h.iloc[-2]<=0: return "BULLISH"
    if h.iloc[-1]<0 and h.iloc[-2]>=0: return "BEARISH"
    if h.iloc[-1]>h.iloc[-2]>h.iloc[-3]: return "BULLISH_TREND"
    if h.iloc[-1]<h.iloc[-2]<h.iloc[-3]: return "BEARISH_TREND"
    return "NEUTRAL"

def rsi_calc(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(span=period,adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=period,adjust=False).mean()
    return float((100-100/(1+g/l.replace(0,np.nan))).iloc[-1])

# ─── SCORING (mirror scan()) ─────────────────────────────────────────────────────
def score_bar(window, spy_window):
    """window/spy_window: DataFrames ending at the bar being evaluated."""
    close, high, low, open_, vol = window.Close, window.High, window.Low, window.Open, window.Volume
    score = 0

    direction, dpts = confirmed_donchian(close); score += dpts
    sdir, spts = ema_stack(close)
    if sdir == direction and spts > 0: score += spts

    adx, dip, dim = adx_calc(high, low, close)
    if adx >= MIN_ADX and ((direction=="BULLISH" and dip>dim) or (direction=="BEARISH" and dim>dip)): score += 1
    if bb_squeeze(close): score += 1

    md = macd_dir(close)
    if (direction=="BULLISH" and md in ("BULLISH","BULLISH_TREND")) or (direction=="BEARISH" and md in ("BEARISH","BEARISH_TREND")): score += 1

    vavg = float(vol.iloc[-21:-1].mean()); vr = float(vol.iloc[-1])/vavg if vavg>0 else 1.0
    if vr >= 1.5: score += 1

    bull_c, bear_c = candle_quality(close.iloc[-1], open_.iloc[-1], high.iloc[-1], low.iloc[-1])
    if (direction=="BULLISH" and bull_c) or (direction=="BEARISH" and bear_c): score += 1

    # relative strength vs SPY (20-bar)
    sret = (float(close.iloc[-1])-float(close.iloc[-21]))/float(close.iloc[-21])*100
    spyret = (float(spy_window.Close.iloc[-1])-float(spy_window.Close.iloc[-21]))/float(spy_window.Close.iloc[-21])*100
    if (direction=="BULLISH" and sret>spyret) or (direction=="BEARISH" and sret<spyret): score += 1

    aok, _ = atr_regime(high, low, close)
    if aok: score += 1

    # SPY regime
    spy_bull = float(spy_window.Close.iloc[-1]) > float(ema(spy_window.Close,50).iloc[-1])
    if (direction=="BULLISH" and spy_bull) or (direction=="BEARISH" and not spy_bull): score += 1

    rsi = rsi_calc(close)
    # No upper cap for calls / lower cap for puts — momentum continuation outperforms.
    if (direction=="BULLISH" and rsi>=40) or (direction=="BEARISH" and rsi<=60): score += 1

    score += 1  # no-earnings assumed (can't check historically here)
    return direction, score, round(adx,1), round(rsi,1)

# ─── RUN BACKTEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*64}\n  BACKTEST: {SYMBOL} — last {EVAL_DAYS} trading days (Swing mode)\n{'='*64}")
    print(f"  Rules: signal at close if score≥{MIN_SCORE} & ADX≥{MIN_ADX}")
    print(f"  Trade: enter next open, hold ≤{HOLD_BARS}d, stop -{STOP_PCT}% / target +{TGT_PCT}% (underlying)\n")

    data = yf.Ticker(SYMBOL).history(period="2y")[["Open","High","Low","Close","Volume"]]
    spy  = yf.Ticker("SPY").history(period="2y")[["Open","High","Low","Close","Volume"]]
    data.index = data.index.tz_localize(None); spy.index = spy.index.tz_localize(None)
    common = data.index.intersection(spy.index)
    data, spy = data.loc[common], spy.loc[common]

    if len(data) < 220:
        print("Not enough history."); sys.exit()

    eval_start = len(data) - EVAL_DAYS
    trades = []
    for i in range(eval_start, len(data)):
        window     = data.iloc[:i+1]
        spy_window = spy.iloc[:i+1]
        if len(window) < 220: continue
        direction, score, adx, rsi = score_bar(window, spy_window)
        sig_date = data.index[i].date()
        if direction == "NEUTRAL" or score < MIN_SCORE:
            print(f"  {sig_date}  —  no signal (dir={direction}, score={score}/13)")
            continue

        if i+1 >= len(data):
            print(f"  {sig_date}  ⚡ {('CALL' if direction=='BULLISH' else 'PUT')} score={score}/13 — too recent to evaluate outcome")
            continue
        entry = float(data.Open.iloc[i+1])
        outcome, exit_px, held = "open", None, 0
        peak = entry  # best favorable price reached
        for j in range(i+1, min(i+1+HOLD_BARS, len(data))):
            held += 1
            hi, lo = float(data.High.iloc[j]), float(data.Low.iloc[j])
            if direction == "BULLISH":
                if lo <= entry*(1-STOP_PCT/100): outcome, exit_px = "STOP", entry*(1-STOP_PCT/100); break
                peak = max(peak, hi)
                if peak >= entry*(1+TRAIL_ARM/100) and lo <= peak*(1-TRAIL_PCT/100):
                    outcome, exit_px = "TRAIL", peak*(1-TRAIL_PCT/100); break
                if hi >= entry*(1+TGT_PCT/100):  outcome, exit_px = "TARGET", entry*(1+TGT_PCT/100); break
            else:
                if hi >= entry*(1+STOP_PCT/100): outcome, exit_px = "STOP", entry*(1+STOP_PCT/100); break
                peak = min(peak, lo)
                if peak <= entry*(1-TRAIL_ARM/100) and hi >= peak*(1+TRAIL_PCT/100):
                    outcome, exit_px = "TRAIL", peak*(1+TRAIL_PCT/100); break
                if lo <= entry*(1-TGT_PCT/100):  outcome, exit_px = "TARGET", entry*(1-TGT_PCT/100); break
        if exit_px is None:
            exit_px = float(data.Close.iloc[min(i+HOLD_BARS, len(data)-1)]); outcome = "TIME"
        ret = (exit_px-entry)/entry*100 * (1 if direction=="BULLISH" else -1)
        trades.append(ret)
        tag = "CALL" if direction=="BULLISH" else "PUT"
        print(f"  {sig_date}  🎯 {tag} score={score}/13 adx={adx} rsi={rsi} | entry {entry:.2f} → {outcome} {exit_px:.2f} ({held}d) = {ret:+.2f}% underlying")

    print(f"\n{'-'*64}")
    if trades:
        wins = [t for t in trades if t > 0]
        print(f"  Signals taken : {len(trades)}")
        print(f"  Win rate      : {len(wins)}/{len(trades)} = {len(wins)/len(trades)*100:.0f}%")
        print(f"  Avg return    : {np.mean(trades):+.2f}% (underlying, per trade)")
        print(f"  Best / Worst  : {max(trades):+.2f}% / {min(trades):+.2f}%")
        print(f"  Total         : {sum(trades):+.2f}% (underlying)")
        print(f"\n  Note: a 30-DTE option would roughly 3–6x these moves (and the")
        print(f"  losers too). This measures the SIGNAL's directional edge only.")
    else:
        print("  No qualifying signals in the window.")
    print(f"{'='*64}\n")
