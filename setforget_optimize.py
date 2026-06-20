#!/usr/bin/env python3
"""
Does folding AlexG's rules into Top-Down OPTIMIZE it? Head-to-head, stock, daily.

BASELINE  (current bot): 50-bar level reclaim + 200-EMA trend + HV>=40, 2:1.
ENHANCED  (+ AlexG):     same, PLUS the level must be a multi-tested AOI (3+ pivot
                         touches) AND a quality reversal candle (body>40% of range,
                         closed back across the level). Tested at 2:1 and 3:1.

Compares fill count, win%, expectancy, and year-by-year consistency.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

NAMES = ["AMD","GOOGL","ARM","HOOD","QQQ","MSTR","AMZN","COIN","SPY","PLTR","NVDA","TSLA",
         "MU","META","SOFI","AVGO","SMCI","AAPL","MSFT","NFLX","CVNA","MARA","RIOT","AFRM"]
K=3; TOL=0.025; LOOKBACK=120; HOLD=40; SLIP=0.10
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def hv(c,n=20):
    r=np.log(c/c.shift()).dropna(); return float(r.rolling(n).std().iloc[-1]*np.sqrt(252)*100)

def pivots(df):
    L,H=df.Low.values,df.High.values; lows=[];highs=[]
    for i in range(K,len(df)-K):
        if L[i]==min(L[i-K:i+K+1]): lows.append(i)
        if H[i]==max(H[i-K:i+K+1]): highs.append(i)
    return lows,highs

def gen(df, enhanced):
    e200=ema(df.Close,200).values; e21=ema(df.Close,21).values
    O,H,L,C=df.Open.values,df.High.values,df.Low.values,df.Close.values
    sl=df.Low.rolling(50).min().shift(1).values; sh=df.High.rolling(50).max().shift(1).values
    lows,highs=pivots(df); lpx=[L[i] for i in lows]; hpx=[H[i] for i in highs]
    out=[]
    hv_series = (np.log(df.Close/df.Close.shift()).rolling(20).std()*np.sqrt(252)*100).values
    for i in range(200,len(df)-1):
        if np.isnan(hv_series[i]) or hv_series[i]<40: continue
        up=C[i]>e200[i]; dn=C[i]<e200[i]
        rng=H[i]-L[i]
        if rng<=0: continue
        if up and L[i]<sl[i] and C[i]>sl[i] and C[i]>O[i]:
            if enhanced:
                touches=sum(1 for k,p in enumerate(lpx) if lows[k]<i and i-lows[k]<=LOOKBACK and abs(p-sl[i])/sl[i]<TOL)
                quality=(C[i]-O[i])>0.4*rng
                if touches<3 or not quality: continue
            out.append((i,"BULLISH",min(L[i],sl[i])))
        elif dn and H[i]>sh[i] and C[i]<sh[i] and C[i]<O[i]:
            if enhanced:
                touches=sum(1 for k,p in enumerate(hpx) if highs[k]<i and i-highs[k]<=LOOKBACK and abs(p-sh[i])/sh[i]<TOL)
                quality=(O[i]-C[i])>0.4*rng
                if touches<3 or not quality: continue
            out.append((i,"BEARISH",max(H[i],sh[i])))
    return out

def backtest(enhanced, rr):
    rows=[]
    for sym in NAMES:
        try:
            df=yf.Ticker(sym).history(period="5y")[["Open","High","Low","Close"]]
            if len(df)<300: continue
            df.index=df.index.tz_localize(None); n=len(df)
            for (i,d,stop_ref) in gen(df, enhanced):
                if i+1>=len(df): continue
                entry=float(df.Open.iloc[i+1]); bull=d=="BULLISH"
                risk=(entry-stop_ref) if bull else (stop_ref-entry)
                if risk<=0: continue
                stop=stop_ref; tgt=entry+rr*risk if bull else entry-rr*risk
                ex=None
                for k in range(i+1,min(i+1+HOLD,len(df))):
                    Hh=float(df.High.iloc[k]); Ll=float(df.Low.iloc[k])
                    if bull:
                        if Ll<=stop: ex=stop; break
                        if Hh>=tgt: ex=tgt; break
                    else:
                        if Hh>=stop: ex=stop; break
                        if Ll<=tgt: ex=tgt; break
                if ex is None: ex=float(df.Close.iloc[min(i+HOLD,len(df)-1)])
                rows.append(((n-1-i)//252,(ex-entry)/entry*100*(1 if bull else -1)-SLIP))
        except Exception: pass
    return rows

def show(label, rows):
    if not rows: print(f"  {label}: no signals"); return
    u=np.array([x for _,x in rows]); yrs=sorted(set(y for y,_ in rows))
    pos=sum(1 for yb in yrs if np.mean([x for y,x in rows if y==yb])>0)
    print(f"  {label:<26} N={len(u):>4} win {np.mean(u>0)*100:>3.0f}% exp {u.mean():>+6.3f}% total {u.sum():>+7.0f}% | profitable {pos}/{len(yrs)} yrs")

if __name__=="__main__":
    print(f"\n{'='*78}\n  OPTIMIZE? Top-Down BASELINE vs +AlexG enhancements (daily stock, 5yr)\n{'='*78}")
    for rr in [2.0, 3.0]:
        print(f"\n  --- {rr:.0f}:1 target ---")
        show(f"Baseline TopDown {rr:.0f}:1", backtest(False, rr))
        show(f"+AlexG enhanced {rr:.0f}:1", backtest(True, rr))
    print()
