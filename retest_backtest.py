#!/usr/bin/env python3
"""
Does Ochoa's RETEST limit entry beat a market entry? (1h Top-Down, stock)

Market entry : buy next bar's open (always trades).
Retest entry : place a limit at the signal-candle MIDPOINT; fill only if a later
               bar (within ~6h, like a DAY order) trades through it — else NO trade.

We compare fill rate, win rate, and expectancy, both at ideal fills and with
0.3% stop slippage. High-vol liquid names, ~2yr of 1h bars.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

NAMES = ["AMD","NVDA","MU","COIN","TSLA","MSTR","HOOD","PLTR","MARA","RIOT","SOFI","AFRM","CVNA","APP","META"]
HOLD = 26; FILL_WINDOW = 6; BASE_SLIP = 0.10
def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def signals(df):
    e=ema(df.Close,200); lo=df.Low.rolling(50).min().shift(1); hi=df.High.rolling(50).max().shift(1)
    out=[]
    for i in range(200,len(df)-1):
        c=float(df.Close.iloc[i]); o=float(df.Open.iloc[i]); l=float(df.Low.iloc[i]); h=float(df.High.iloc[i])
        sl=float(lo.iloc[i]); sh=float(hi.iloc[i]); up=c>e.iloc[i]
        if up and l<sl and c>sl and c>o:   out.append((i,"BULLISH",min(l,sl),(h+l)/2))
        elif (not up) and h>sh and c<sh and c<o: out.append((i,"BEARISH",max(h,sh),(h+l)/2))
    return out

def play(df, j_entry, entry, bull, sref, stop_slip):
    risk=(entry-sref) if bull else (sref-entry)
    if risk<=0: risk=entry*0.01
    stop=entry-risk if bull else entry+risk
    tgt=entry+2*risk if bull else entry-2*risk
    ex=None
    for j in range(j_entry, min(j_entry+HOLD, len(df))):
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
    mkt=[]; rt=[]; rt_signals=0
    for sym in NAMES:
        try:
            df=yf.Ticker(sym).history(period="730d",interval="1h")[["Open","High","Low","Close"]]
            if len(df)<600: continue
            df.index=pd.to_datetime(df.index)
            for (i,d,sref,mid) in signals(df):
                if i+1>=len(df): continue
                bull=d=="BULLISH"
                # MARKET entry
                mkt.append(play(df, i+1, float(df.Open.iloc[i+1]), bull, sref, stop_slip))
                # RETEST entry: limit at midpoint, fill if a bar trades through within window
                rt_signals+=1; filled=None
                for j in range(i+1, min(i+1+FILL_WINDOW, len(df))):
                    if bull and float(df.Low.iloc[j])<=mid: filled=j; break
                    if (not bull) and float(df.High.iloc[j])>=mid: filled=j; break
                if filled is not None:
                    rt.append(play(df, filled, mid, bull, sref, stop_slip))
        except Exception: pass
    return np.array(mkt), np.array(rt), rt_signals

if __name__=="__main__":
    print(f"\n{'='*64}\n  MARKET vs RETEST (limit) entry — 1h Top-Down stock\n{'='*64}")
    for ss in [0.0, 0.3]:
        m, r, sig = backtest(ss)
        print(f"\n  --- stop slippage {ss}% ---")
        print(f"  {'Entry':<10}{'Signals':>8}{'Fills':>7}{'Fill%':>7}{'Win%':>7}{'Avg%':>9}{'Total%':>10}")
        print(f"  {'Market':<10}{len(m):>8}{len(m):>7}{'100%':>7}{np.mean(m>0)*100:>6.0f}%{m.mean():>+8.3f}%{m.sum():>+9.0f}%")
        fr = len(r)/sig*100 if sig else 0
        if len(r):
            print(f"  {'Retest':<10}{sig:>8}{len(r):>7}{fr:>6.0f}%{np.mean(r>0)*100:>6.0f}%{r.mean():>+8.3f}%{r.sum():>+9.0f}%")
    print(f"\n{'='*64}\n")
