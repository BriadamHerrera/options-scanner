#!/usr/bin/env python3
"""
Best-effort MECHANICAL PROXY of the top-down multi-timeframe framework:

  1. Key level on higher TF   → 50-day swing high/low (proxy for 4H/1H structure)
  2. Wait for price to reach it
  3/4. Confirmation = RECLAIM (sweep through the level then close back across it)
       + higher-timeframe TREND alignment (above/below 200 EMA)
  5. Risk-first: stop beyond the level, 2:1 target

CANNOT capture: 'absorption', real-time order-flow reading, discretionary
'shift of structure'. So this tests the SKELETON, not a skilled trader's read.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import smc_test as smc

NAMES = smc.NAMES
SWING = 50          # "higher timeframe" key level
TOL = 0.005

def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def signals(df):
    e200 = ema(df.Close, 200)
    lo = df.Low.rolling(SWING).min().shift(1)
    hi = df.High.rolling(SWING).max().shift(1)
    sig = pd.Series(index=df.index, dtype=object); info={}
    for i in range(SWING+1, len(df)):
        c=float(df.Close.iloc[i]); o=float(df.Open.iloc[i])
        l=float(df.Low.iloc[i]); h=float(df.High.iloc[i])
        sl=float(lo.iloc[i]); sh=float(hi.iloc[i]); trend_up = c > e200.iloc[i]
        # RECLAIM of support, ALIGNED with uptrend, bullish candle
        if trend_up and l < sl and c > sl and c > o:
            sig.iloc[i]="BULLISH"; info[i]=min(l, sl)
        # RECLAIM of resistance, aligned with downtrend
        elif (not trend_up) and h > sh and c < sh and c < o:
            sig.iloc[i]="BEARISH"; info[i]=max(h, sh)
    return sig, info

def sim_rr(df, i, bull, stop_ref, hold=10):
    entry=float(df.Open.iloc[i+1])
    risk = (entry-stop_ref) if bull else (stop_ref-entry)
    if risk<=0: risk=entry*0.02
    stop = entry-risk if bull else entry+risk
    tgt  = entry+2*risk if bull else entry-2*risk
    ex=None; held=0
    for j in range(i+1,min(i+1+hold,len(df))):
        held=j-i; hh=float(df.High.iloc[j]); ll=float(df.Low.iloc[j])
        if bull:
            if ll<=stop: ex=stop; break
            if hh>=tgt: ex=tgt; break
        else:
            if hh>=stop: ex=stop; break
            if ll<=tgt: ex=tgt; break
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
            u,held=sim_rr(df,i,direction=="BULLISH",info[i])
            rows.append(((n-1-i)//252, u, smc.option_ret(u,held)))
    return rows

if __name__=="__main__":
    print(f"\n{'='*66}\n  TOP-DOWN FRAMEWORK (mechanical proxy) — level+reclaim+trend\n{'='*66}")
    print("  Captures: key level, reclaim, HTF trend, 2:1 risk")
    print("  Misses:   absorption, order-flow, discretionary structure read\n")
    rows=backtest()
    if not rows: print("  No signals."); raise SystemExit
    u=np.array([x[1] for x in rows]); o=np.array([x[2] for x in rows])
    print(f"  Total signals: {len(u)}")
    print(f"  UNDERLYING: win {np.mean(u>0)*100:.0f}%  avg {u.mean():+.3f}%/trade  total {u.sum():+.0f}%")
    print(f"  OPTIONS   : win {np.mean(o>0)*100:.0f}%  avg {o.mean():+.2f}%/trade  total {o.sum():+.0f}%")
    print(f"\n  By year:")
    print(f"  {'Years ago':<10}{'N':>5}{'Win%':>7}{'Underlying':>12}{'Option':>10}")
    for yb in sorted(set(y for y,_,_ in rows)):
        uu=np.array([x for y,x,_ in rows if y==yb]); oo=np.array([z for y,_,z in rows if y==yb])
        print(f"  {yb:<10}{len(uu):>5}{np.mean(uu>0)*100:>6.0f}%{uu.mean():>+11.2f}%{oo.mean():>+9.2f}%")
    posu=sum(1 for yb in set(y for y,_,_ in rows) if np.mean([x for y,x,_ in rows if y==yb])>0)
    poso=sum(1 for yb in set(y for y,_,_ in rows) if np.mean([z for y,_,z in rows if y==yb])>0)
    print(f"\n  Underlying profitable {posu}/4 years | Options profitable {poso}/4 years")
    print(f"{'='*66}\n")
