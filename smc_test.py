#!/usr/bin/env python3
"""
Smart Money Concepts — honest test of the two most testable ChartPrime tools:
  A) LIQUIDITY SWEEP REVERSAL — price wicks below an N-bar swing low (grabs
     sell-side liquidity / stops), then CLOSES back above it → long. Mirror for short.
  B) FAIR VALUE GAP (FVG) FILL — a 3-candle imbalance; enter when price returns
     to the gap, betting it continues in the gap's direction.

Both run through the same rigor: 6 rolling windows + realistic option costs.
Definitions are fixed a priori (not tuned to maximize results).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

NAMES = ["AMD","GOOGL","ARM","HOOD","QQQ","MSTR","AMZN","COIN","SPY",
         "PLTR","NVDA","TSLA","MU","META","SOFI","AVGO","SMCI"]
LEVERAGE, THETA_DAY, SPREAD_RT = 5.0, 0.7, 4.0
HOLD_BARS, STOP_PCT = 8, 4.0
SWING = 10           # swing lookback for liquidity levels
def option_ret(u, held): return LEVERAGE*u - THETA_DAY*held - SPREAD_RT

def simulate(df, i, bull, target_mode="opp"):
    """Enter next open, exit on stop / target / time. Returns (uret, held)."""
    entry=float(df.Open.iloc[i+1]); ex=None; held=0
    for j in range(i+1, min(i+1+HOLD_BARS, len(df))):
        held=j-i; hi,lo,cl=float(df.High.iloc[j]),float(df.Low.iloc[j]),float(df.Close.iloc[j])
        if bull:
            if lo<=entry*(1-STOP_PCT/100): ex=entry*(1-STOP_PCT/100); break
            if hi>=entry*(1+2*STOP_PCT/100): ex=entry*(1+2*STOP_PCT/100); break  # 2:1 target
        else:
            if hi>=entry*(1+STOP_PCT/100): ex=entry*(1+STOP_PCT/100); break
            if lo<=entry*(1-2*STOP_PCT/100): ex=entry*(1-2*STOP_PCT/100); break
    if ex is None: ex=float(df.Close.iloc[min(i+HOLD_BARS,len(df)-1)]); held=HOLD_BARS
    return (ex-entry)/entry*100*(1 if bull else -1), max(held,1)

def liquidity_sweep_signals(df):
    """BULLISH: low wicks below prior SWING-bar low but close > that low (sweep & reclaim)."""
    sig=pd.Series(index=df.index, dtype=object)
    swing_lo = df.Low.rolling(SWING).min().shift(1)
    swing_hi = df.High.rolling(SWING).max().shift(1)
    for i in range(SWING+1, len(df)):
        lo,hi,cl=float(df.Low.iloc[i]),float(df.High.iloc[i]),float(df.Close.iloc[i])
        sl,sh=float(swing_lo.iloc[i]),float(swing_hi.iloc[i])
        if lo < sl and cl > sl:    sig.iloc[i]="BULLISH"   # swept sell-side, reclaimed
        elif hi > sh and cl < sh:  sig.iloc[i]="BEARISH"   # swept buy-side, rejected
    return sig

def fvg_signals(df):
    """Bullish FVG at bar i: low[i] > high[i-2] (gap up imbalance) → momentum long when it forms."""
    sig=pd.Series(index=df.index, dtype=object)
    for i in range(2, len(df)):
        if float(df.Low.iloc[i]) > float(df.High.iloc[i-2]):   sig.iloc[i]="BULLISH"
        elif float(df.High.iloc[i]) < float(df.Low.iloc[i-2]): sig.iloc[i]="BEARISH"
    return sig

def backtest(signal_fn):
    """Return list of (years_ago, option_ret)."""
    out=[]
    for sym in NAMES:
        df=yf.Ticker(sym).history(period="5y")[["Open","High","Low","Close","Volume"]]
        if len(df)<800: continue
        df.index=df.index.tz_localize(None)
        sig=signal_fn(df); n=len(df)
        for i in range(260, len(df)-1):
            direction=sig.iloc[i]
            if direction not in ("BULLISH","BEARISH"): continue
            u,held=simulate(df,i,direction=="BULLISH")
            out.append(((n-1-i)//252, option_ret(u,held)))
    return out

def report(name, rows):
    print(f"\n  {name}")
    print(f"  {'Years ago':<10}{'N':>6}{'Win%':>8}{'OPT exp':>10}{'OPT total':>12}")
    allr=[r for _,r in rows]
    for yb in sorted(set(y for y,_ in rows)):
        a=np.array([r for y,r in rows if y==yb])
        print(f"  {yb:<10}{len(a):>6}{np.mean(a>0)*100:>7.0f}%{a.mean():>+9.2f}%{a.sum():>+11.1f}%")
    a=np.array(allr)
    pos_years=sum(1 for yb in set(y for y,_ in rows) if np.mean([r for y,r in rows if y==yb])>0)
    tot_years=len(set(y for y,_ in rows))
    print(f"  {'-'*46}")
    print(f"  {'ALL':<10}{len(a):>6}{np.mean(a>0)*100:>7.0f}%{a.mean():>+9.2f}%{a.sum():>+11.1f}%")
    print(f"  Profitable in {pos_years}/{tot_years} years after costs.")

if __name__=="__main__":
    print(f"\n{'='*66}\n  SMART MONEY CONCEPTS — honest multi-window + cost test\n{'='*66}")
    print(f"  Option model: {LEVERAGE}x lev − {THETA_DAY}%/day theta − {SPREAD_RT}% spread")
    report("A) LIQUIDITY SWEEP REVERSAL", backtest(liquidity_sweep_signals))
    report("B) FAIR VALUE GAP (momentum)", backtest(fvg_signals))
    print(f"\n{'='*66}\n")
