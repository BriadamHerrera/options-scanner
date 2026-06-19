#!/usr/bin/env python3
"""
Measure the Volatility Risk Premium (VRP) — the actual source of edge for
premium-SELLING strategies.

VRP = Implied Volatility (what option BUYERS pay) − Realized Volatility (what
the stock actually did). If IV > RV systematically, option SELLERS are paid
that gap on average. This is the one structural, documented edge in options.

We pull live ~30 DTE ATM implied vol from option chains and compare to recent
realized vol, across the watchlist.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from datetime import datetime

NAMES = ["SPY","QQQ","AMD","GOOGL","ARM","HOOD","MSTR","AMZN","COIN",
         "PLTR","NVDA","TSLA","MU","META","AAPL","MSFT"]

def realized_vol(close, days=21):
    r = np.log(close/close.shift()).dropna()
    return float(r.iloc[-days:].std()*np.sqrt(252)*100)

def atm_iv(tk, spot):
    try:
        exps = tk.options
        if not exps: return None
        today = datetime.today().date()
        best, bd = None, 999
        for e in exps:
            dte = (datetime.strptime(e,"%Y-%m-%d").date()-today).days
            if 20 <= dte <= 45 and abs(dte-30) < bd: bd, best = abs(dte-30), e
        if not best:
            for e in exps:
                dte=(datetime.strptime(e,"%Y-%m-%d").date()-today).days
                if dte>=20 and abs(dte-30)<bd: bd,best=abs(dte-30),e
        if not best: return None
        ch = tk.option_chain(best)
        ivs=[]
        for df in (ch.calls, ch.puts):
            df=df.copy(); df["d"]=(df.strike-spot).abs()
            row=df.sort_values("d").iloc[0]
            iv=float(row.get("impliedVolatility",0) or 0)
            if iv>0: ivs.append(iv*100)
        return np.mean(ivs) if ivs else None
    except Exception:
        return None

if __name__=="__main__":
    print(f"\n{'='*60}\n  VOLATILITY RISK PREMIUM — live IV vs realized\n{'='*60}")
    print(f"  {'Ticker':<7}{'IV%':>8}{'RV%':>8}{'VRP (IV-RV)':>14}")
    print(f"  {'-'*37}")
    vrps=[]
    for sym in NAMES:
        try:
            tk=yf.Ticker(sym)
            h=tk.history(period="3mo")
            if h.empty: continue
            spot=float(h.Close.iloc[-1]); rv=realized_vol(h.Close)
            iv=atm_iv(tk, spot)
            if iv is None:
                print(f"  {sym:<7}{'—':>8}{rv:>8.1f}{'no chain':>14}"); continue
            vrp=iv-rv; vrps.append(vrp)
            flag = "✓ sell" if vrp>0 else "✗"
            print(f"  {sym:<7}{iv:>8.1f}{rv:>8.1f}{vrp:>+11.1f}  {flag}")
        except Exception:
            continue
    print(f"  {'-'*37}")
    if vrps:
        v=np.array(vrps)
        print(f"  Average VRP: {v.mean():+.1f} vol points")
        print(f"  Positive in {np.mean(v>0)*100:.0f}% of names")
        print(f"\n  IV > RV means option BUYERS overpaid → SELLERS collect the gap.")
    print(f"{'='*60}\n")
