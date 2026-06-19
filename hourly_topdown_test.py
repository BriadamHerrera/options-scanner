#!/usr/bin/env python3
"""
Top-Down concept applied to 1-HOUR bars — more signals, but does the edge hold?

Same logic as the daily version, scaled down:
  • Key level   = 50-hour swing high/low (~1.5 weeks of structure)
  • HTF trend   = 200-period EMA on 1h (~30 trading days) — trade WITH it
  • Reclaim     = 1h candle closes back across the level
  • Risk        = 2:1, stop beyond the level
  • Hold        = up to ~4 trading days (26 hours)

Honest accounting: MORE signals also means MORE cost drag. Option model applied
by converting the hold duration to days. High-vol watchlist only.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

NAMES = ["SMCI","ARM","MU","AMD","HOOD","MARA","MSTR","CVNA","APP","COIN",
         "PLTR","NVDA","TSLA","RDDT","SOFI","RIOT","AFRM"]
LEV, THETA_DAY, SPREAD = 5.0, 0.4, 3.0     # 30-DTE option model
HOLD_HOURS = 26
BARS_PER_DAY = 6.5

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def opt(u, hours): return LEV*u - THETA_DAY*(hours/BARS_PER_DAY) - SPREAD

def signals(df):
    e = ema(df.Close, 200)
    lo = df.Low.rolling(50).min().shift(1)
    hi = df.High.rolling(50).max().shift(1)
    sig = pd.Series(index=df.index, dtype=object); info={}
    for i in range(200, len(df)):
        c=float(df.Close.iloc[i]); o=float(df.Open.iloc[i])
        l=float(df.Low.iloc[i]); h=float(df.High.iloc[i])
        sl=float(lo.iloc[i]); sh=float(hi.iloc[i]); up=c>e.iloc[i]
        if up and l<sl and c>sl and c>o:
            sig.iloc[i]="BULLISH"; info[i]=min(l,sl)
        elif (not up) and h>sh and c<sh and c<o:
            sig.iloc[i]="BEARISH"; info[i]=max(h,sh)
    return sig, info

def backtest():
    rows=[]
    for sym in NAMES:
        try:
            df=yf.Ticker(sym).history(period="730d", interval="1h")[["Open","High","Low","Close"]]
            if len(df) < 600: continue
            df.index = pd.to_datetime(df.index)
            sig,info=signals(df)
            for i in range(200, len(df)-1):
                d=sig.iloc[i]
                if d not in ("BULLISH","BEARISH"): continue
                entry=float(df.Open.iloc[i+1]); bull=d=="BULLISH"; sref=info[i]
                risk=(entry-sref) if bull else (sref-entry)
                if risk<=0: risk=entry*0.01
                stop=entry-risk if bull else entry+risk
                tgt=entry+2*risk if bull else entry-2*risk
                ex=None; hh=0
                for j in range(i+1, min(i+1+HOLD_HOURS, len(df))):
                    hh=j-i; H=float(df.High.iloc[j]); L=float(df.Low.iloc[j])
                    if bull:
                        if L<=stop: ex=stop; break
                        if H>=tgt: ex=tgt; break
                    else:
                        if H>=stop: ex=stop; break
                        if L<=tgt: ex=tgt; break
                if ex is None: ex=float(df.Close.iloc[min(i+HOLD_HOURS,len(df)-1)]); hh=HOLD_HOURS
                u=(ex-entry)/entry*100*(1 if bull else -1)
                rows.append((u, opt(u, max(hh,1))))
        except Exception as e:
            print(f"  {sym}: {e}")
    return rows

if __name__=="__main__":
    print(f"\n{'='*60}\n  TOP-DOWN on 1-HOUR bars (~2yr) — does it hold?\n{'='*60}")
    rows=backtest()
    if not rows: print("  No data."); raise SystemExit
    u=np.array([x[0] for x in rows]); o=np.array([x[1] for x in rows])
    print(f"  Signals: {len(u)}  (vs ~76 on daily over 5yr — far more)")
    print(f"  UNDERLYING: win {np.mean(u>0)*100:.0f}%  avg {u.mean():+.3f}%/trade  total {u.sum():+.0f}%")
    print(f"  OPTIONS   : win {np.mean(o>0)*100:.0f}%  avg {o.mean():+.2f}%/trade  total {o.sum():+.0f}%")
    print(f"{'='*60}\n")
