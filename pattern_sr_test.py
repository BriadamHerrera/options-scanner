#!/usr/bin/env python3
"""
Reversal patterns at NOTABLE (previously-tested) S/R on 4h & 1h — PURE STOCK.

Your refinement:
  • Level must have been TESTED BEFORE — a prior pivot sits at the same price
    (within tol) before the pattern forms → "notable" S/R, not a random touch.
  • Double Bottom / Double Top forms AT that level.
  • Enter only on CONFIRMATION (neckline break). Stop beyond the level, 2:1.
  • Pure STOCK returns, minus realistic slippage (entry + stop). No options.

Tested on 1h and 4h (~2yr), with a year-by-year split.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

NAMES = ["AMD","NVDA","MU","COIN","TSLA","MSTR","HOOD","PLTR","MARA","RIOT",
         "SOFI","AFRM","CVNA","APP","META","SMCI","ARM","NFLX","AMZN","GOOGL"]
K=3; LEVEL_TOL=0.025; PRIOR_LOOKBACK=150; HOLD=30
BASE_SLIP=0.10; STOP_SLIP=0.30

def to_4h(df):
    g=df.reset_index(drop=True); g["grp"]=g.index//4
    return g.groupby("grp").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).reset_index(drop=True)

def pivots(df):
    L,H=df.Low.values,df.High.values; lows=[];highs=[]
    for i in range(K,len(df)-K):
        if L[i]==min(L[i-K:i+K+1]): lows.append(i)
        if H[i]==max(H[i-K:i+K+1]): highs.append(i)
    return lows,highs

def tested_before(price, idx, piv_idx, piv_px, lookback):
    """True if an earlier pivot sits near `price` before bar idx (notable level)."""
    for k, pi in enumerate(piv_idx):
        if pi >= idx: break
        if idx - pi <= lookback and abs(piv_px[k]-price)/price < LEVEL_TOL:
            return True
    return False

def find(df):
    lows,highs=pivots(df)
    L,H,C=df.Low.values,df.High.values,df.Close.values
    lpx=[L[i] for i in lows]; hpx=[H[i] for i in highs]
    sigs=[]
    # double bottom at notable support
    for a in range(len(lows)-1):
        i1,i2=lows[a],lows[a+1]
        if not (5<=i2-i1<=60): continue
        if abs(L[i1]-L[i2])/L[i1]>LEVEL_TOL: continue
        if not tested_before(L[i2], i1, lows+highs, lpx+hpx, PRIOR_LOOKBACK): continue
        mids=[h for h in highs if i1<h<i2]
        if not mids: continue
        neck=max(H[h] for h in mids)
        for j in range(i2+1,min(i2+25,len(df))):
            if C[j]>neck: sigs.append((j,"BULLISH",min(L[i1],L[i2]))); break
    # double top at notable resistance
    for a in range(len(highs)-1):
        i1,i2=highs[a],highs[a+1]
        if not (5<=i2-i1<=60): continue
        if abs(H[i1]-H[i2])/H[i1]>LEVEL_TOL: continue
        if not tested_before(H[i2], i1, lows+highs, lpx+hpx, PRIOR_LOOKBACK): continue
        mids=[l for l in lows if i1<l<i2]
        if not mids: continue
        neck=min(L[l] for l in mids)
        for j in range(i2+1,min(i2+25,len(df))):
            if C[j]<neck: sigs.append((j,"BEARISH",max(H[i1],H[i2]))); break
    return sigs

def backtest(tf):
    rows=[]
    for sym in NAMES:
        try:
            df=yf.Ticker(sym).history(period="730d",interval="1h")[["Open","High","Low","Close"]]
            if len(df)<600: continue
            df=df.reset_index(drop=True)
            if tf=="4h": df=to_4h(df)
            n=len(df)
            for (j,d,sref) in find(df):
                if j+1>=len(df): continue
                entry=float(df.Open.iloc[j+1]); bull=d=="BULLISH"
                risk=(entry-sref) if bull else (sref-entry)
                if risk<=0: continue
                stop=entry-risk if bull else entry+risk; tgt=entry+2*risk if bull else entry-2*risk
                ex=None
                for k in range(j+1,min(j+1+HOLD,len(df))):
                    Hh=float(df.High.iloc[k]); Ll=float(df.Low.iloc[k])
                    if bull:
                        if Ll<=stop: ex=stop*(1-STOP_SLIP/100); break
                        if Hh>=tgt: ex=tgt; break
                    else:
                        if Hh>=stop: ex=stop*(1+STOP_SLIP/100); break
                        if Ll<=tgt: ex=tgt; break
                if ex is None: ex=float(df.Close.iloc[min(j+HOLD,len(df)-1)])
                u=(ex-entry)/entry*100*(1 if bull else -1)-BASE_SLIP
                bars_per_year = 252*6.5 if tf=="1h" else 252*1.6
                rows.append((int((n-1-j)/bars_per_year), u))
        except Exception: pass
    return rows

if __name__=="__main__":
    for tf in ["1h","4h"]:
        rows=backtest(tf)
        print(f"\n{'='*56}\n  Notable-S/R reversal patterns — {tf} STOCK (slip incl.)\n{'='*56}")
        if not rows: print("  No patterns."); continue
        u=np.array([x for _,x in rows])
        print(f"  Patterns: {len(u)} | Win {np.mean(u>0)*100:.0f}% | Avg {u.mean():+.3f}% | Total {u.sum():+.0f}%")
        print(f"  {'Yrs ago':<8}{'N':>5}{'Win%':>7}{'Avg%':>9}{'Total%':>10}")
        pos=0; yrs=sorted(set(y for y,_ in rows))
        for yb in yrs:
            a=np.array([x for y,x in rows if y==yb])
            if a.mean()>0: pos+=1
            print(f"  {yb:<8}{len(a):>5}{np.mean(a>0)*100:>6.0f}%{a.mean():>+8.3f}%{a.sum():>+9.1f}%")
        print(f"  Profitable in {pos}/{len(yrs)} years")
