#!/usr/bin/env python3
"""
OUT-OF-SAMPLE test — the honest one.

We optimized parameters on the LAST 252 trading days (in-sample).
This runs the SAME optimized parameters on the PRIOR 252 trading days
(out-of-sample) — data the strategy was never tuned on. If the edge holds
here, it's real. If it collapses, it was curve-fitting.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import backtest as bt
import mr_backtest as mr

TREND_NAMES = ["AMD","GOOGL","ARM","HOOD","QQQ","MSTR","AMZN","COIN","SPY"]
MR_NAMES    = ["PLTR","NVDA","TSLA","MU","META","SOFI","AVGO","SMCI"]

# Optimized exit params (must match the app)
HOLD, STOP, TRAIL, ARM, TGT = 10, 4.0, 4.0, 4.0, 12.0
MR_HOLD, MR_STOP = 7, 5.0
TREND_FLOOR, MR_FLOOR = 9, 6

def trade_trend(d, s, i):
    direction, score, _, _ = bt.score_bar(d.iloc[:i+1], s.iloc[:i+1])
    if direction=="NEUTRAL" or score<TREND_FLOOR or i+1>=len(d): return None
    entry=float(d.Open.iloc[i+1]); ex=None; peak=entry; bull=direction=="BULLISH"
    for j in range(i+1, min(i+1+HOLD, len(d))):
        hi,lo=float(d.High.iloc[j]),float(d.Low.iloc[j])
        if bull:
            if lo<=entry*(1-STOP/100): ex=entry*(1-STOP/100); break
            peak=max(peak,hi)
            if peak>=entry*(1+ARM/100) and lo<=peak*(1-TRAIL/100): ex=peak*(1-TRAIL/100); break
            if hi>=entry*(1+TGT/100): ex=entry*(1+TGT/100); break
        else:
            if hi>=entry*(1+STOP/100): ex=entry*(1+STOP/100); break
            peak=min(peak,lo)
            if peak<=entry*(1-ARM/100) and hi>=peak*(1+TRAIL/100): ex=peak*(1+TRAIL/100); break
            if lo<=entry*(1-TGT/100): ex=entry*(1-TGT/100); break
    if ex is None: ex=float(d.Close.iloc[min(i+HOLD,len(d)-1)])
    return (ex-entry)/entry*100*(1 if bull else -1)

def trade_mr(d, i):
    direction, score, _, _ = mr.mr_score(d.iloc[:i+1])
    if direction=="NEUTRAL" or score<MR_FLOOR or i+1>=len(d): return None
    entry=float(d.Open.iloc[i+1]); ex=None; bull=direction=="BULLISH"
    for j in range(i+1, min(i+1+MR_HOLD, len(d))):
        hi,lo,cl=float(d.High.iloc[j]),float(d.Low.iloc[j]),float(d.Close.iloc[j])
        mean_now=float((d.Close.iloc[:j+1].rolling(10).mean()).iloc[-1])
        if bull:
            if lo<=entry*(1-MR_STOP/100): ex=entry*(1-MR_STOP/100); break
            if cl>=mean_now: ex=cl; break
        else:
            if hi>=entry*(1+MR_STOP/100): ex=entry*(1+MR_STOP/100); break
            if cl<=mean_now: ex=cl; break
    if ex is None: ex=float(d.Close.iloc[min(i+MR_HOLD,len(d)-1)])
    return (ex-entry)/entry*100*(1 if bull else -1)

def run(names, strat, win):
    """win: ('in', 'oos'). Returns list of trade returns."""
    spy=yf.Ticker("SPY").history(period="5y")[["Open","High","Low","Close","Volume"]]
    spy.index=spy.index.tz_localize(None)
    out=[]
    for sym in names:
        d=yf.Ticker(sym).history(period="5y")[["Open","High","Low","Close","Volume"]]
        d.index=d.index.tz_localize(None)
        c=d.index.intersection(spy.index); d,s=d.loc[c],spy.loc[c]
        n=len(d)
        if n<800: continue
        rng = range(n-252, n) if win=="in" else range(n-504, n-252)
        for i in rng:
            if i<260: continue
            r = trade_trend(d,s,i) if strat=="trend" else trade_mr(d,i)
            if r is not None: out.append(r)
    return out

def summary(label, arr):
    if not arr:
        print(f"  {label:<28} no trades"); return
    a=np.array(arr)
    print(f"  {label:<28} N={len(a):<4} win={np.mean(a>0)*100:>3.0f}%  exp={a.mean():>+5.2f}%  total={a.sum():>+7.1f}%")

if __name__ == "__main__":
    print(f"\n{'='*70}\n  OUT-OF-SAMPLE VALIDATION\n{'='*70}")
    print("  In-sample  = last 252 trading days (where we optimized)")
    print("  Out-sample = the 252 days BEFORE that (never tuned on)\n")

    print("  TREND strategy (robust trender watchlist)")
    summary("  In-sample (optimized)",  run(TREND_NAMES,"trend","in"))
    summary("  OUT-OF-SAMPLE",          run(TREND_NAMES,"trend","oos"))

    print("\n  MEAN-REVERSION strategy (choppy names)")
    summary("  In-sample (optimized)",  run(MR_NAMES,"mean_reversion","in"))
    summary("  OUT-OF-SAMPLE",          run(MR_NAMES,"mean_reversion","oos"))
    print(f"{'='*70}\n")
