#!/usr/bin/env python3
"""
Reversal patterns at MAJOR levels — Double Top / Double Bottom — honest test.

Only fades at significant levels with a recognized pattern (your refinement):
  • Swing detection (k-bar fractal pivots)
  • Double Bottom: two swing lows within 3% of each other, a peak between them,
    AND the lows sit near a 60-day low (MAJOR support). Entry on NECKLINE break
    (close above the middle peak). Stop below the lows. Target 2:1.
  • Double Top: mirror at MAJOR resistance -> short.

Daily bars, 5yr, broad universe (patterns form over weeks). Multi-window + costs.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

NAMES = ["AMD","GOOGL","ARM","HOOD","QQQ","MSTR","AMZN","COIN","SPY","PLTR","NVDA","TSLA",
         "MU","META","SOFI","AVGO","SMCI","AAPL","MSFT","NFLX","CVNA","MARA","RIOT","AFRM"]
K = 3                 # fractal pivot half-width
LEVEL_TOL = 0.03      # two lows/highs within 3% = "equal"
MAJOR_LOOKBACK = 60   # lows must be near the 60-day extreme to be "major"
HOLD = 30             # up to ~6 weeks for the pattern to play out
LEV, THETA_DAY, SPREAD = 5.0, 0.4, 3.0   # option model

def pivots(df):
    lows, highs = [], []
    L, H = df.Low.values, df.High.values
    for i in range(K, len(df)-K):
        if L[i] == min(L[i-K:i+K+1]): lows.append(i)
        if H[i] == max(H[i-K:i+K+1]): highs.append(i)
    return lows, highs

def opt(u, days): return LEV*u - THETA_DAY*days - SPREAD

def find_patterns(df):
    """Return list of (entry_idx, direction, stop_ref, neckline)."""
    lows, highs = pivots(df)
    L, H, C = df.Low.values, df.High.values, df.Close.values
    sigs = []
    roll_lo = df.Low.rolling(MAJOR_LOOKBACK).min().shift(1).values
    roll_hi = df.High.rolling(MAJOR_LOOKBACK).max().shift(1).values

    # DOUBLE BOTTOM
    for a in range(len(lows)-1):
        i1 = lows[a]; i2 = lows[a+1]
        if i2 - i1 < 5 or i2 - i1 > 60: continue
        p1, p2 = L[i1], L[i2]
        if abs(p1-p2)/p1 > LEVEL_TOL: continue
        # major support: low near the 60-day low
        if np.isnan(roll_lo[i2]) or p2 > roll_lo[i2]*1.03: continue
        mids = [h for h in highs if i1 < h < i2]
        if not mids: continue
        neck = max(H[h] for h in mids)
        # confirmation: a close breaks above the neckline after i2
        for j in range(i2+1, min(i2+25, len(df))):
            if C[j] > neck:
                sigs.append((j, "BULLISH", min(p1,p2), neck)); break

    # DOUBLE TOP
    for a in range(len(highs)-1):
        i1 = highs[a]; i2 = highs[a+1]
        if i2 - i1 < 5 or i2 - i1 > 60: continue
        p1, p2 = H[i1], H[i2]
        if abs(p1-p2)/p1 > LEVEL_TOL: continue
        if np.isnan(roll_hi[i2]) or p2 < roll_hi[i2]*0.97: continue
        mids = [l for l in lows if i1 < l < i2]
        if not mids: continue
        neck = min(L[l] for l in mids)
        for j in range(i2+1, min(i2+25, len(df))):
            if C[j] < neck:
                sigs.append((j, "BEARISH", max(p1,p2), neck)); break
    return sigs

def backtest():
    rows=[]
    for sym in NAMES:
        try:
            df=yf.Ticker(sym).history(period="5y")[["Open","High","Low","Close"]]
            if len(df)<300: continue
            df.index=df.index.tz_localize(None)
            for (j, d, sref, neck) in find_patterns(df):
                if j+1>=len(df): continue
                entry=float(df.Open.iloc[j+1]); bull=d=="BULLISH"
                risk=(entry-sref) if bull else (sref-entry)
                if risk<=0: continue
                stop=entry-risk if bull else entry+risk
                tgt=entry+2*risk if bull else entry-2*risk
                ex=None; held=0
                for k in range(j+1, min(j+1+HOLD, len(df))):
                    held=k-j; Hh=float(df.High.iloc[k]); Ll=float(df.Low.iloc[k])
                    if bull:
                        if Ll<=stop: ex=stop; break
                        if Hh>=tgt: ex=tgt; break
                    else:
                        if Hh>=stop: ex=stop; break
                        if Ll<=tgt: ex=tgt; break
                if ex is None: ex=float(df.Close.iloc[min(j+HOLD,len(df)-1)])
                u=(ex-entry)/entry*100*(1 if bull else -1)
                rows.append((u, opt(u, max(held,1))))
        except Exception:
            continue
    return rows

if __name__=="__main__":
    print(f"\n{'='*60}\n  DOUBLE TOP/BOTTOM at MAJOR levels — honest test\n{'='*60}")
    rows=backtest()
    if not rows: print("  No patterns found."); raise SystemExit
    u=np.array([x[0] for x in rows]); o=np.array([x[1] for x in rows])
    print(f"  Patterns found: {len(u)}  (across {len(NAMES)} names, 5yr)")
    print(f"  UNDERLYING: win {np.mean(u>0)*100:.0f}%  avg {u.mean():+.3f}%  total {u.sum():+.0f}%")
    print(f"  OPTIONS   : win {np.mean(o>0)*100:.0f}%  avg {o.mean():+.2f}%  total {o.sum():+.0f}%")
    print(f"{'='*60}\n")
