#!/usr/bin/env python3
"""
Mean-Reversion strategy + backtest — the inverse of the breakout system.

Logic: buy oversold dips in uptrends, fade overbought rips in downtrends.
Works best on choppy / range-bound names where breakouts fail (TSLA, PLTR...).

Scoring (0–10):
  1. RSI(2) extreme (<10 bull / >90 bear)        → +2  (core Connors signal)
  2. Outside Bollinger Band (2.5 std)            → +2
  3. Z-score of close vs SMA20 beyond ±2         → +1
  4. ADX < 25 (choppy regime — MR works here)    → +1
  5. RSI(14) < 30 / > 70                          → +1
  6. 3+ consecutive down/up closes               → +1
  7. With longer trend (dip ABOVE SMA200 = buy,  → +1   ("buy the dip in an uptrend")
     rip BELOW SMA200 = fade)
  8. Volume capitulation spike (>1.5x)           → +1

Exit: target = revert to SMA10 (the mean) | stop = -5% adverse | time = 7 days
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from datetime import datetime

SYMBOL    = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
EVAL_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 252
MR_MIN_SCORE = 6
HOLD_BARS = 7
STOP_PCT  = 5.0     # knife protection

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def sma(s,n): return s.rolling(n).mean()

def rsi(close, period):
    d = close.diff()
    g = d.clip(lower=0).ewm(span=period,adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=period,adjust=False).mean()
    return 100 - 100/(1+g/l.replace(0,np.nan))

def adx_calc(high,low,close,period=14):
    tr = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(span=period,adjust=False).mean()
    dmp=(high-high.shift()).clip(lower=0); dmm=(low.shift()-low).clip(lower=0)
    dmp=dmp.where(dmp>dmm,0); dmm=dmm.where(dmm>dmp,0)
    dip=100*dmp.ewm(span=period,adjust=False).mean()/atr
    dim=100*dmm.ewm(span=period,adjust=False).mean()/atr
    dx=100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return float(dx.ewm(span=period,adjust=False).mean().iloc[-1])

def mr_score(window):
    """Return (direction, score, rsi2, adx) for mean-reversion at last bar."""
    close, high, low, vol = window.Close, window.High, window.Low, window.Volume
    c = float(close.iloc[-1])
    sma20 = float(sma(close,20).iloc[-1]); std20 = float(close.rolling(20).std().iloc[-1])
    sma200 = float(sma(close,200).iloc[-1]) if len(close)>=200 else sma20
    if std20 == 0: return "NEUTRAL",0,50,0
    z = (c - sma20)/std20
    rsi2 = float(rsi(close,2).iloc[-1]); rsi14 = float(rsi(close,14).iloc[-1])
    adx = adx_calc(high,low,close)

    # direction: oversold dip in uptrend = BULLISH; overbought rip in downtrend = BEARISH
    oversold   = rsi2 < 10 or z < -2
    overbought = rsi2 > 90 or z > 2
    if oversold and c > sma200:   direction = "BULLISH"
    elif overbought and c < sma200: direction = "BEARISH"
    elif oversold:                direction = "BULLISH"   # dip even without trend filter (weaker)
    elif overbought:              direction = "BEARISH"
    else:                         return "NEUTRAL",0,rsi2,adx

    score = 0
    if (direction=="BULLISH" and rsi2<10) or (direction=="BEARISH" and rsi2>90): score += 2
    lower = sma20 - 2.5*std20; upper = sma20 + 2.5*std20
    if (direction=="BULLISH" and c<lower) or (direction=="BEARISH" and c>upper): score += 2
    if abs(z) >= 2: score += 1
    if adx < 25: score += 1
    if (direction=="BULLISH" and rsi14<30) or (direction=="BEARISH" and rsi14>70): score += 1
    # consecutive closes
    diffs = close.diff().iloc[-3:]
    if direction=="BULLISH" and (diffs<0).all(): score += 1
    if direction=="BEARISH" and (diffs>0).all(): score += 1
    if (direction=="BULLISH" and c>sma200) or (direction=="BEARISH" and c<sma200): score += 1
    vavg=float(vol.iloc[-21:-1].mean()); vr=float(vol.iloc[-1])/vavg if vavg>0 else 1
    if vr>=1.5: score += 1
    return direction, score, round(rsi2,1), round(adx,1)


if __name__ == "__main__":
    print(f"\n{'='*64}\n  MEAN-REVERSION BACKTEST: {SYMBOL} — {EVAL_DAYS} trading days\n{'='*64}")
    print(f"  Buy oversold dips / fade overbought rips | score≥{MR_MIN_SCORE}")
    print(f"  Exit: revert to SMA10 | stop -{STOP_PCT}% | time {HOLD_BARS}d\n")
    d = yf.Ticker(SYMBOL).history(period="2y")[["Open","High","Low","Close","Volume"]]
    d.index = d.index.tz_localize(None)
    if len(d) < 230: print("Not enough history."); sys.exit()
    trades = []
    for i in range(len(d)-EVAL_DAYS, len(d)):
        if i < 210 or i+1 >= len(d): continue
        w = d.iloc[:i+1]
        direction, score, rsi2, adx = mr_score(w)
        if direction=="NEUTRAL" or score < MR_MIN_SCORE: continue
        entry = float(d.Open.iloc[i+1]); exit_px, outcome = None, "TIME"
        bull = direction=="BULLISH"
        for j in range(i+1, min(i+1+HOLD_BARS, len(d))):
            hi,lo,cl = float(d.High.iloc[j]),float(d.Low.iloc[j]),float(d.Close.iloc[j])
            mean_now = float(sma(d.Close.iloc[:j+1],10).iloc[-1])
            if bull:
                if lo <= entry*(1-STOP_PCT/100): exit_px,outcome=entry*(1-STOP_PCT/100),"STOP"; break
                if cl >= mean_now: exit_px,outcome=cl,"MEAN"; break
            else:
                if hi >= entry*(1+STOP_PCT/100): exit_px,outcome=entry*(1+STOP_PCT/100),"STOP"; break
                if cl <= mean_now: exit_px,outcome=cl,"MEAN"; break
        if exit_px is None: exit_px=float(d.Close.iloc[min(i+HOLD_BARS,len(d)-1)])
        ret=(exit_px-entry)/entry*100*(1 if bull else -1)
        trades.append(ret)
        tag="CALL" if bull else "PUT"
        print(f"  {d.index[i].date()}  {tag} sc={score}/10 rsi2={rsi2} adx={adx} | {entry:.2f}→{outcome} {exit_px:.2f} = {ret:+.2f}%")
    print(f"\n{'-'*64}")
    if trades:
        a=np.array(trades); w=a[a>0]
        print(f"  Signals {len(a)} | Win {len(w)/len(a)*100:.0f}% | Avg {a.mean():+.2f}% | Total {a.sum():+.1f}%")
    else:
        print("  No signals.")
    print(f"{'='*64}\n")
