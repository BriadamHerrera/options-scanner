#!/usr/bin/env python3
"""
Whale-Strategy Scanner + Backtest (free-data version).

The institutional-flow signal can't be replicated without a paid feed, and has
NO historical data, so it CANNOT be backtested. What we CAN do:

  SCANNER (live): use call/put options VOLUME as a flow proxy + trend (20-SMA)
                  + IV check + ATM structure → ranked call ideas. Runnable now.

  BACKTEST: test the tradeable CORE (the part that's historically measurable):
            'bullish proxy (volume surge + up day) while above 20-SMA → buy a
            ~30-DTE ATM call, 2:1 target.' Option-cost adjusted, year-by-year.
            This tests the ENGINE (trend-following), NOT the whale signal.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np, pandas as pd, yfinance as yf
from datetime import datetime

WATCH = ["AMD","NVDA","MU","COIN","TSLA","MSTR","HOOD","PLTR","MARA","RIOT",
         "SOFI","AFRM","CVNA","APP","META","AMZN","GOOGL","AAPL","MSFT","NFLX"]

# option-cost model (30-DTE ATM): leverage, theta %/day, round-trip spread %
LEV, THETA, SPREAD = 5.0, 0.4, 3.0
HOLD = 10; STOP_PCT = 4.0; TGT_PCT = 8.0   # 2:1


# ─── LIVE SCANNER (flow proxy via current options volume) ────────────────────────
def flow_proxy(tk):
    """Call vs put volume across the nearest 2 expiries = crude bullish-flow proxy."""
    try:
        exps = tk.options[:2]
        cv = pv = 0
        for e in exps:
            ch = tk.option_chain(e)
            cv += int(ch.calls["volume"].fillna(0).sum())
            pv += int(ch.puts["volume"].fillna(0).sum())
        ratio = cv / max(pv, 1)
        return cv, pv, ratio
    except Exception:
        return 0, 0, 0.0

def iv_rank_proxy(close):
    r = np.log(close/close.shift()).dropna()
    rv = r.rolling(20).std()*np.sqrt(252)*100
    cur = rv.iloc[-1]; lo, hi = rv.tail(252).quantile(0.05), rv.tail(252).quantile(0.95)
    return float(np.clip((cur-lo)/(hi-lo)*100, 0, 100)) if hi > lo else 50.0

def scan():
    print(f"\n{'='*72}\n  WHALE-STRATEGY SCANNER (live, flow PROXY)  {datetime.now():%Y-%m-%d %H:%M}\n{'='*72}")
    print("  Proxy: call/put volume ratio | Trend: price>20SMA | IV Rank<70 | bullish only")
    print(f"  {'Ticker':<7}{'Price':>9}{'C/P vol':>9}{'vs20SMA':>9}{'IVrank':>8}{'Verdict':>14}")
    print(f"  {'-'*56}")
    rows=[]
    for s in WATCH:
        try:
            tk = yf.Ticker(s)
            h = tk.history(period="6mo")
            if h.empty: continue
            close = h["Close"]; price = float(close.iloc[-1])
            sma20 = float(close.rolling(20).mean().iloc[-1])
            cv, pv, ratio = flow_proxy(tk)
            ivr = iv_rank_proxy(close)
            bullish = ratio > 1.5
            uptrend = price > sma20
            iv_ok = ivr <= 70
            verdict = "🟢 CANDIDATE" if (bullish and uptrend and iv_ok) else "—"
            print(f"  {s:<7}{price:>9.2f}{ratio:>9.2f}{'↑' if uptrend else '↓':>9}{ivr:>7.0f}%{verdict:>14}")
            if verdict.startswith("🟢"):
                rows.append((s, price, ratio, ivr))
        except Exception as e:
            print(f"  {s:<7} err {e}")
    print(f"  {'-'*56}")
    if rows:
        print(f"\n  {len(rows)} candidates (bullish flow proxy + uptrend + reasonable IV):")
        for s,p,r,iv in sorted(rows, key=lambda x:-x[2]):
            atm = round(p/5)*5
            print(f"    {s}: buy ~30-45 DTE ${atm} call (ATM) | C/P vol {r:.1f} | IVrank {iv:.0f}%")
    else:
        print("\n  No candidates right now.")


# ─── BACKTEST (the tradeable core: trend + volume proxy → ATM call) ───────────────
def backtest():
    print(f"\n{'='*72}\n  BACKTEST — strategy CORE (trend + volume-surge proxy → ATM call)\n{'='*72}")
    print("  ⚠️ Tests the ENGINE only — the institutional-flow signal has no")
    print("     historical data and CANNOT be backtested. This is the baseline the")
    print("     whale signal would have to BEAT to be worth anything.\n")
    rows=[]
    for s in WATCH:
        try:
            d = yf.Ticker(s).history(period="5y")[["Open","High","Low","Close","Volume"]]
            if len(d) < 300: continue
            d.index = d.index.tz_localize(None); n=len(d)
            c = d["Close"]; sma20 = c.rolling(20).mean(); vavg = d["Volume"].rolling(20).mean()
            for i in range(220, len(d)-1):
                # bullish proxy: above 20SMA + up day + volume surge (stand-in for "bullish flow")
                above = c.iloc[i] > sma20.iloc[i]
                up_day = c.iloc[i] > d["Open"].iloc[i]
                vol_surge = d["Volume"].iloc[i] > 1.3*vavg.iloc[i]
                if not (above and up_day and vol_surge): continue
                entry = float(d["Open"].iloc[i+1]); ex=None; held=0
                for j in range(i+1, min(i+1+HOLD, len(d))):
                    held=j-i; hi,lo=float(d["High"].iloc[j]),float(d["Low"].iloc[j])
                    if lo <= entry*(1-STOP_PCT/100): ex=entry*(1-STOP_PCT/100); break
                    if hi >= entry*(1+TGT_PCT/100):  ex=entry*(1+TGT_PCT/100); break
                if ex is None: ex=float(d["Close"].iloc[min(i+HOLD,len(d)-1)]); held=HOLD
                u=(ex-entry)/entry*100
                opt = LEV*u - THETA*held - SPREAD          # option-cost adjusted
                rows.append(((n-1-i)//252, u, opt))
        except Exception: pass
    if not rows: print("  no signals."); return
    u=np.array([x[1] for x in rows]); o=np.array([x[2] for x in rows])
    print(f"  Signals: {len(u)}")
    print(f"  UNDERLYING: win {np.mean(u>0)*100:.0f}%  avg {u.mean():+.3f}%  total {u.sum():+.0f}%")
    print(f"  OPTIONS   : win {np.mean(o>0)*100:.0f}%  avg {o.mean():+.2f}%  total {o.sum():+.0f}%")
    print(f"\n  By year (option P/L):")
    print(f"  {'Yrs ago':<8}{'N':>5}{'Win%':>7}{'Undrly':>9}{'Option':>9}")
    pos=0; yrs=sorted(set(y for y,_,_ in rows))
    for yb in yrs:
        uu=np.array([x for y,x,_ in rows if y==yb]); oo=np.array([z for y,_,z in rows if y==yb])
        if oo.mean()>0: pos+=1
        print(f"  {yb:<8}{len(uu):>5}{np.mean(uu>0)*100:>6.0f}%{uu.mean():>+8.2f}%{oo.mean():>+8.2f}%")
    print(f"  Profitable (options) in {pos}/{len(yrs)} years")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        scan()
    else:
        backtest()
