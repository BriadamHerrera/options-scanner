#!/usr/bin/env python3
"""
AlexG "Set and Forget" — mechanical core, honest test (stock proxy).

Rules implemented:
  • Trend filter: close > rising 21-EMA (uptrend) / < falling for downtrend
  • AOI = level with 3+ pivot touches (multi-tested support/resistance)
  • Pullback to the AOI + bullish/bearish reversal candle (engulfing/close-back)
  • SET & FORGET: fixed stop beyond the AOI, fixed R:R target, no management
Tested at 2:1 and 3:1 targets, daily bars 5yr, with year-by-year split.
(Original is forex; tested on our stock universe — principles transfer.)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

NAMES = ["AMD","GOOGL","ARM","HOOD","QQQ","MSTR","AMZN","COIN","SPY","PLTR","NVDA","TSLA",
         "MU","META","SOFI","AVGO","SMCI","AAPL","MSFT","NFLX","CVNA","MARA"]
K=3; TOL=0.025; LOOKBACK=120; HOLD=40
def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def pivots(df):
    L,H=df.Low.values,df.High.values; lows=[];highs=[]
    for i in range(K,len(df)-K):
        if L[i]==min(L[i-K:i+K+1]): lows.append(i)
        if H[i]==max(H[i-K:i+K+1]): highs.append(i)
    return lows,highs

def signals(df):
    e21=ema(df.Close,21); e21v=e21.values
    O,H,L,C=df.Open.values,df.High.values,df.Low.values,df.Close.values
    lows,highs=pivots(df); lpx=[L[i] for i in lows]; hpx=[H[i] for i in highs]
    sigs=[]
    for i in range(60,len(df)-1):
        up = C[i]>e21v[i] and e21v[i]>e21v[i-20]
        dn = C[i]<e21v[i] and e21v[i]<e21v[i-20]
        # bullish: uptrend, price pulled back to an AOI with 3+ prior low-touches, reversal candle
        if up:
            # find an AOI support near current low with >=3 touches
            near=[p for k,p in enumerate(lpx) if lows[k]<i and i-lows[k]<=LOOKBACK and abs(p-L[i])/L[i]<TOL]
            touches=len(near)
            bull_candle = C[i]>O[i] and (C[i]-O[i])>0.4*(H[i]-L[i]) and L[i]<=min(near)*(1+TOL) if near else False
            if touches>=3 and bull_candle:
                sigs.append((i,"BULLISH", min(near)*(1-0.005)))   # stop just below AOI
        elif dn:
            near=[p for k,p in enumerate(hpx) if highs[k]<i and i-highs[k]<=LOOKBACK and abs(p-H[i])/H[i]<TOL]
            touches=len(near)
            bear_candle = C[i]<O[i] and (O[i]-C[i])>0.4*(H[i]-L[i]) and H[i]>=max(near)*(1-TOL) if near else False
            if touches>=3 and bear_candle:
                sigs.append((i,"BEARISH", max(near)*(1+0.005)))
    return sigs

def backtest(rr):
    rows=[]
    for sym in NAMES:
        try:
            df=yf.Ticker(sym).history(period="5y")[["Open","High","Low","Close"]]
            if len(df)<300: continue
            df.index=df.index.tz_localize(None); n=len(df)
            for (i,d,stop_ref) in signals(df):
                if i+1>=len(df): continue
                entry=float(df.Open.iloc[i+1]); bull=d=="BULLISH"
                risk=(entry-stop_ref) if bull else (stop_ref-entry)
                if risk<=0: continue
                stop=stop_ref; tgt=entry+rr*risk if bull else entry-rr*risk
                ex=None
                for k in range(i+1,min(i+1+HOLD,len(df))):  # set & forget: only stop or target
                    Hh=float(df.High.iloc[k]); Ll=float(df.Low.iloc[k])
                    if bull:
                        if Ll<=stop: ex=stop; break
                        if Hh>=tgt: ex=tgt; break
                    else:
                        if Hh>=stop: ex=stop; break
                        if Ll<=tgt: ex=tgt; break
                if ex is None: ex=float(df.Close.iloc[min(i+HOLD,len(df)-1)])
                u=(ex-entry)/entry*100*(1 if bull else -1)-0.10
                rows.append(((n-1-i)//252, u))
        except Exception: pass
    return rows

if __name__=="__main__":
    for rr in [2.0, 3.0]:
        rows=backtest(rr)
        print(f"\n{'='*56}\n  AlexG Set & Forget — {rr:.0f}:1 target, STOCK, daily\n{'='*56}")
        if not rows: print("  No signals."); continue
        u=np.array([x for _,x in rows])
        print(f"  Signals: {len(u)} | Win {np.mean(u>0)*100:.0f}% | Avg {u.mean():+.3f}% | Total {u.sum():+.0f}%")
        pos=0; yrs=sorted(set(y for y,_ in rows))
        for yb in yrs:
            a=np.array([x for y,x in rows if y==yb])
            if a.mean()>0: pos+=1
            print(f"  {yb} yr ago: N={len(a):>3} win {np.mean(a>0)*100:>3.0f}% avg {a.mean():>+6.3f}% total {a.sum():>+7.1f}%")
        print(f"  Profitable in {pos}/{len(yrs)} years")
