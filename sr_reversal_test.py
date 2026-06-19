#!/usr/bin/env python3
"""
KEY SUPPORT/RESISTANCE REVERSAL on bigger timeframes.

Unlike the 10-bar liquidity-sweep test, this uses MAJOR levels (20/50/100-day
highs & lows = key swing structure) and requires a rejection candle:
  • Support bounce (BULLISH): price dips to/below the N-day low but CLOSES back
    above it on a bullish candle.
  • Resistance reject (BEARISH): price pokes the N-day high but CLOSES back below
    on a bearish candle.

Optionally require the level to have been TOUCHED multiple times (more "key").
Run through multi-window + option costs.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import smc_test as smc   # reuse simulate() + option_ret()

NAMES = smc.NAMES
TOL = 0.01   # within 1% counts as "touching" the level

def sr_signals(df, lookback):
    sig=pd.Series(index=df.index, dtype=object)
    lo = df.Low.rolling(lookback).min().shift(1)
    hi = df.High.rolling(lookback).max().shift(1)
    for i in range(lookback+1, len(df)):
        c=float(df.Close.iloc[i]); o=float(df.Open.iloc[i])
        l=float(df.Low.iloc[i]); h=float(df.High.iloc[i])
        sl=float(lo.iloc[i]); sh=float(hi.iloc[i])
        # support bounce: touched the level, closed back above, bullish candle
        if l <= sl*(1+TOL) and c > sl and c > o:
            sig.iloc[i]="BULLISH"
        # resistance reject
        elif h >= sh*(1-TOL) and c < sh and c < o:
            sig.iloc[i]="BEARISH"
    return sig

def backtest(lookback):
    rows=[]
    for sym in NAMES:
        df=yf.Ticker(sym).history(period="5y")[["Open","High","Low","Close","Volume"]]
        if len(df)<800: continue
        df.index=df.index.tz_localize(None)
        sig=sr_signals(df, lookback); n=len(df)
        for i in range(260,len(df)-1):
            direction=sig.iloc[i]
            if direction not in ("BULLISH","BEARISH"): continue
            u,held=smc.simulate(df,i,direction=="BULLISH")
            rows.append(((n-1-i)//252, smc.option_ret(u,held), u))
    return rows

if __name__=="__main__":
    print(f"\n{'='*68}\n  KEY S/R REVERSAL (bigger timeframes) — multi-window + costs\n{'='*68}")
    for lb in (20, 50, 100):
        rows=backtest(lb)
        if not rows:
            print(f"\n  {lb}-day levels: no signals"); continue
        u=np.array([x[2] for x in rows]); o=np.array([x[1] for x in rows])
        yrs=sorted(set(y for y,_,_ in rows))
        pos=sum(1 for yb in yrs if np.mean([x for y,x,_ in rows if y==yb])>0)
        print(f"\n  {lb}-DAY KEY LEVELS  (N={len(o)})")
        print(f"    Underlying avg : {u.mean():+.3f}%/trade")
        print(f"    Option avg     : {o.mean():+.2f}%/trade   win {np.mean(o>0)*100:.0f}%")
        print(f"    Profitable in  : {pos}/{len(yrs)} years")
    print(f"\n{'='*68}\n")
