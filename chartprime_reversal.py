#!/usr/bin/env python3
"""
ChartPrime-style REVERSAL signal — faithful reimplementation + honest OOS test.

ChartPrime's reversal arrows are built on a Supertrend (ATR trailing-stop) flip,
typically with an oscillator confirmation. This reconstructs that concept:

  • Supertrend(period=10, mult=3) trailing stop
  • A reversal BUY fires when the trend flips up; SELL when it flips down
  • Optional RSI confirmation (ChartPrime often gates signals with momentum)

Then it runs the SAME in-sample vs out-of-sample test we used on our own
strategies — the test ChartPrime (and every indicator vendor) never publishes.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

TREND_NAMES = ["AMD","GOOGL","ARM","HOOD","QQQ","MSTR","AMZN","COIN","SPY"]
MR_NAMES    = ["PLTR","NVDA","TSLA","MU","META","SOFI","AVGO","SMCI"]
ALL_NAMES   = TREND_NAMES + MR_NAMES

ATR_PERIOD, ATR_MULT = 7, 2.0   # more responsive — fires like ChartPrime's arrows
USE_RSI_FILTER = True
HOLD_BARS = 10          # cap hold; primary exit is the opposite flip
STOP_PCT  = 5.0

def atr(df, period):
    h,l,c = df.High, df.Low, df.Close
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def rsi(close, period=14):
    d=close.diff()
    g=d.clip(lower=0).ewm(span=period,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(span=period,adjust=False).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def supertrend(df, period=None, mult=None):
    """Returns a Series of +1 (uptrend) / -1 (downtrend) and the flip points."""
    period = ATR_PERIOD if period is None else period
    mult   = ATR_MULT   if mult   is None else mult
    hl2 = (df.High+df.Low)/2
    a = atr(df, period)
    upper = hl2 + mult*a
    lower = hl2 - mult*a
    fu = upper.copy(); fl = lower.copy()
    dir_ = pd.Series(index=df.index, dtype=float)
    close = df.Close
    for i in range(len(df)):
        if i==0:
            dir_.iloc[i]=1; continue
        # final upper band
        fu.iloc[i] = min(upper.iloc[i], fu.iloc[i-1]) if close.iloc[i-1] <= fu.iloc[i-1] else upper.iloc[i]
        fl.iloc[i] = max(lower.iloc[i], fl.iloc[i-1]) if close.iloc[i-1] >= fl.iloc[i-1] else lower.iloc[i]
        # direction
        if close.iloc[i] > fu.iloc[i-1]:   dir_.iloc[i]=1
        elif close.iloc[i] < fl.iloc[i-1]: dir_.iloc[i]=-1
        else:                              dir_.iloc[i]=dir_.iloc[i-1]
    return dir_

def signals(df):
    """Reversal direction at each bar: 'BULLISH' on flip up, 'BEARISH' on flip down, else None."""
    d = supertrend(df)
    r = rsi(df.Close)
    flip_up = (d==1) & (d.shift()==-1)
    flip_dn = (d==-1) & (d.shift()==1)
    out = pd.Series(index=df.index, dtype=object)
    for i in range(len(df)):
        if flip_up.iloc[i]:
            if not USE_RSI_FILTER or r.iloc[i] < 60: out.iloc[i]="BULLISH"
        elif flip_dn.iloc[i]:
            if not USE_RSI_FILTER or r.iloc[i] > 40: out.iloc[i]="BEARISH"
    return out, supertrend(df)

def backtest(names, win):
    spy_unused=None
    all_ret=[]
    for sym in names:
        df = yf.Ticker(sym).history(period="5y")[["Open","High","Low","Close","Volume"]]
        if len(df) < 800: continue
        df.index = df.index.tz_localize(None)
        sig, d = signals(df)
        n=len(df)
        rng = range(n-252, n) if win=="in" else range(n-504, n-252)
        for i in rng:
            if i<260 or i+1>=len(df): continue
            direction = sig.iloc[i]
            if direction not in ("BULLISH","BEARISH"): continue
            entry=float(df.Open.iloc[i+1]); bull=direction=="BULLISH"; ex=None
            for j in range(i+1, min(i+1+HOLD_BARS, len(df))):
                hi,lo=float(df.High.iloc[j]),float(df.Low.iloc[j])
                # exit on opposite supertrend flip
                if (bull and d.iloc[j]==-1) or (not bull and d.iloc[j]==1):
                    ex=float(df.Close.iloc[j]); break
                if bull and lo<=entry*(1-STOP_PCT/100): ex=entry*(1-STOP_PCT/100); break
                if not bull and hi>=entry*(1+STOP_PCT/100): ex=entry*(1+STOP_PCT/100); break
            if ex is None: ex=float(df.Close.iloc[min(i+HOLD_BARS,len(df)-1)])
            all_ret.append((ex-entry)/entry*100*(1 if bull else -1))
    return all_ret

def summary(label, arr):
    if not arr: print(f"  {label:<26} no trades"); return
    a=np.array(arr)
    print(f"  {label:<26} N={len(a):<4} win={np.mean(a>0)*100:>3.0f}%  exp={a.mean():>+5.2f}%  total={a.sum():>+7.1f}%")

if __name__ == "__main__":
    print(f"\n{'='*70}\n  CHARTPRIME-STYLE REVERSAL (Supertrend flip) — honest OOS test\n{'='*70}")
    print(f"  Supertrend({ATR_PERIOD},{ATR_MULT}) + RSI gate | exit on opposite flip / -{STOP_PCT}% stop\n")
    summary("In-sample (last year)",  backtest(ALL_NAMES,"in"))
    summary("OUT-OF-SAMPLE",          backtest(ALL_NAMES,"oos"))
    print(f"{'='*70}\n")
