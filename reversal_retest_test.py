#!/usr/bin/env python3
"""
Does the RETEST entry rescue a REVERSAL strategy? (the right tool for the job)

Reversal setup (1h): price dips to/below a 50h key support and CLOSES back above
it (rejection candle) -> fade long. Mirror at resistance -> fade short.
This is trend-AGNOSTIC fading (the thing the retest entry is designed for).

Compare MARKET entry vs RETEST limit entry (at the rejection-candle midpoint),
on the underlying, with realistic slippage. High-vol liquid names, ~2yr 1h.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

NAMES = ["AMD","NVDA","MU","COIN","TSLA","MSTR","HOOD","PLTR","MARA","RIOT","SOFI","AFRM","CVNA","APP","META"]
HOLD=26; FILL_WINDOW=6; BASE_SLIP=0.10

def reversal_signals(df):
    """Rejection at a 50h level. Returns (i, dir, stop_ref, midpoint)."""
    lo=df.Low.rolling(50).min().shift(1); hi=df.High.rolling(50).max().shift(1)
    out=[]
    for i in range(60,len(df)-1):
        c=float(df.Close.iloc[i]); o=float(df.Open.iloc[i]); l=float(df.Low.iloc[i]); h=float(df.High.iloc[i])
        sl=float(lo.iloc[i]); sh=float(hi.iloc[i]); mid=(h+l)/2
        # dipped below support but closed back above, bullish candle -> fade LONG
        if l<sl and c>sl and c>o:   out.append((i,"BULLISH", l, mid))
        # spiked above resistance but closed back below -> fade SHORT
        elif h>sh and c<sh and c<o: out.append((i,"BEARISH", h, mid))
    return out

def play(df, j_entry, entry, bull, sref, stop_slip):
    risk=(entry-sref) if bull else (sref-entry)
    if risk<=0: risk=entry*0.01
    stop=entry-risk if bull else entry+risk
    tgt=entry+2*risk if bull else entry-2*risk
    ex=None
    for j in range(j_entry, min(j_entry+HOLD,len(df))):
        H=float(df.High.iloc[j]); L=float(df.Low.iloc[j])
        if bull:
            if L<=stop: ex=stop*(1-stop_slip/100); break
            if H>=tgt: ex=tgt; break
        else:
            if H>=stop: ex=stop*(1+stop_slip/100); break
            if L<=tgt: ex=tgt; break
    if ex is None: ex=float(df.Close.iloc[min(j_entry+HOLD-1,len(df)-1)])
    return (ex-entry)/entry*100*(1 if bull else -1) - BASE_SLIP

def backtest(stop_slip):
    mkt=[]; rt=[]; sig=0
    for sym in NAMES:
        try:
            df=yf.Ticker(sym).history(period="730d",interval="1h")[["Open","High","Low","Close"]]
            if len(df)<600: continue
            df.index=pd.to_datetime(df.index)
            for (i,d,sref,mid) in reversal_signals(df):
                if i+1>=len(df): continue
                bull=d=="BULLISH"
                mkt.append(play(df,i+1,float(df.Open.iloc[i+1]),bull,sref,stop_slip))
                sig+=1; filled=None
                for j in range(i+1,min(i+1+FILL_WINDOW,len(df))):
                    if bull and float(df.Low.iloc[j])<=mid: filled=j; break
                    if (not bull) and float(df.High.iloc[j])>=mid: filled=j; break
                if filled is not None:
                    # stop must be beyond the level relative to the midpoint entry
                    rt.append(play(df,filled,mid,bull,sref,stop_slip))
        except Exception: pass
    return np.array(mkt), np.array(rt), sig

if __name__=="__main__":
    print(f"\n{'='*64}\n  REVERSAL strategy: MARKET vs RETEST entry (1h, high-vol)\n{'='*64}")
    for ss in [0.0, 0.3]:
        m,r,sig=backtest(ss)
        print(f"\n  --- stop slippage {ss}% ---")
        print(f"  {'Entry':<10}{'Fills':>7}{'Fill%':>7}{'Win%':>7}{'Avg%':>9}{'Total%':>10}")
        if len(m): print(f"  {'Market':<10}{len(m):>7}{'100%':>7}{np.mean(m>0)*100:>6.0f}%{m.mean():>+8.3f}%{m.sum():>+9.0f}%")
        if len(r):
            fr=len(r)/sig*100
            print(f"  {'Retest':<10}{len(r):>7}{fr:>6.0f}%{np.mean(r>0)*100:>6.0f}%{r.mean():>+8.3f}%{r.sum():>+9.0f}%")
    print(f"\n{'='*64}\n")
