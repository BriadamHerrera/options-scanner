#!/usr/bin/env python3
"""
TradingLab-style EMA TREND-PULLBACK strategy — honest test.

Canonical rules (the archetype these channels teach):
  • Uptrend filter: close > 200 EMA  (downtrend: close < 200 EMA)
  • Pullback: price dips to/touches the 20 EMA
  • Trigger: bullish bounce — closes back above the 20 EMA on a bullish candle
  • Stop: below the pullback low | Target: 2x risk (2:1 R:R)

Tested across multiple windows, on the underlying AND with option costs.
A popular blog claims '82% win rate' for this — let's see.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import smc_test as smc

NAMES = smc.NAMES
def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def signals(df):
    e20=ema(df.Close,20); e50=ema(df.Close,50); e200=ema(df.Close,200)
    sig=pd.Series(index=df.index, dtype=object)
    info={}
    for i in range(200, len(df)):
        c=float(df.Close.iloc[i]); o=float(df.Open.iloc[i])
        l=float(df.Low.iloc[i]); h=float(df.High.iloc[i])
        up = c > e200.iloc[i]; dn = c < e200.iloc[i]
        # LONG: uptrend, pullback touched 20EMA, closed back above on bullish candle
        if up and l <= e20.iloc[i] and c > e20.iloc[i] and c > o:
            sig.iloc[i]="BULLISH"; info[i]=l
        elif dn and h >= e20.iloc[i] and c < e20.iloc[i] and c < o:
            sig.iloc[i]="BEARISH"; info[i]=h
    return sig, info

def simulate_rr(df, i, bull, stop_ref, hold=10):
    entry=float(df.Open.iloc[i+1])
    if bull:
        risk=entry-stop_ref
        if risk<=0: risk=entry*0.02
        stop=entry-risk; tgt=entry+2*risk
    else:
        risk=stop_ref-entry
        if risk<=0: risk=entry*0.02
        stop=entry+risk; tgt=entry-2*risk
    ex=None; held=0
    for j in range(i+1, min(i+1+hold,len(df))):
        held=j-i; hi=float(df.High.iloc[j]); lo=float(df.Low.iloc[j])
        if bull:
            if lo<=stop: ex=stop; break
            if hi>=tgt: ex=tgt; break
        else:
            if hi>=stop: ex=stop; break
            if lo<=tgt: ex=tgt; break
    if ex is None: ex=float(df.Close.iloc[min(i+hold,len(df)-1)]); held=hold
    return (ex-entry)/entry*100*(1 if bull else -1), max(held,1)

def backtest():
    rows=[]
    for sym in NAMES:
        df=yf.Ticker(sym).history(period="5y")[["Open","High","Low","Close","Volume"]]
        if len(df)<800: continue
        df.index=df.index.tz_localize(None)
        sig,info=signals(df); n=len(df)
        for i in range(260,len(df)-1):
            direction=sig.iloc[i]
            if direction not in ("BULLISH","BEARISH"): continue
            u,held=simulate_rr(df,i,direction=="BULLISH",info[i])
            rows.append(((n-1-i)//252, u, smc.option_ret(u,held)))
    return rows

if __name__=="__main__":
    print(f"\n{'='*64}\n  TRADINGLAB-STYLE EMA PULLBACK — honest multi-window test\n{'='*64}")
    rows=backtest()
    if not rows:
        print("  No signals."); raise SystemExit
    u=np.array([x[1] for x in rows]); o=np.array([x[2] for x in rows])
    print(f"  Total signals: {len(u)}")
    print(f"  UNDERLYING: win {np.mean(u>0)*100:.0f}%  avg {u.mean():+.3f}%/trade  total {u.sum():+.0f}%")
    print(f"  OPTIONS   : win {np.mean(o>0)*100:.0f}%  avg {o.mean():+.2f}%/trade  total {o.sum():+.0f}%")
    print(f"\n  By year (option P/L):")
    print(f"  {'Years ago':<10}{'N':>5}{'Win%':>7}{'Underlying':>12}{'Option':>10}")
    for yb in sorted(set(y for y,_,_ in rows)):
        uu=np.array([x for y,x,_ in rows if y==yb]); oo=np.array([z for y,_,z in rows if y==yb])
        print(f"  {yb:<10}{len(uu):>5}{np.mean(uu>0)*100:>6.0f}%{uu.mean():>+11.2f}%{oo.mean():>+9.2f}%")
    pos=sum(1 for yb in set(y for y,_,_ in rows) if np.mean([x for y,_,x in rows if y==yb])>0)
    print(f"\n  Underlying win rate: {np.mean(u>0)*100:.0f}% (claim was '82%')")
    print(f"  Profitable (options) in {pos}/{len(set(y for y,_,_ in rows))} years.")
    print(f"{'='*64}\n")
