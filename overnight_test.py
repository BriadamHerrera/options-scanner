#!/usr/bin/env python3
"""
The OVERNIGHT RETURN ANOMALY — a documented effect with a real mechanism.

Claim (Cooper/Cliff/Gulen and others): nearly all of the equity risk premium
is earned OVERNIGHT (prior close -> next open). The INTRADAY session
(open -> close) is historically flat-to-negative.

Mechanism: overnight you can't react to news (gap risk) → risk premium;
ETF creation/redemption and order-flow timing concentrate gains at the open.

We split each day's return into overnight vs intraday, across the universe and
across multiple years. Then we ask the honest question: is it TRADEABLE after
the brutal cost of round-tripping every single day?
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

NAMES = ["SPY","QQQ","AMD","GOOGL","ARM","HOOD","MSTR","AMZN","COIN",
         "PLTR","NVDA","TSLA","MU","META","AAPL","MSFT","AVGO","SMCI"]

def load(sym):
    d = yf.Ticker(sym).history(period="5y")[["Open","High","Low","Close"]]
    if len(d) < 500: return None
    d.index = d.index.tz_localize(None)
    d["overnight"] = (d.Open / d.Close.shift() - 1) * 100      # prior close -> open
    d["intraday"]  = (d.Close / d.Open - 1) * 100              # open -> close
    d["full"]      = (d.Close / d.Close.shift() - 1) * 100
    return d.dropna()

def compound(series_pct):
    return (np.prod(1 + series_pct/100) - 1) * 100

if __name__ == "__main__":
    print(f"\n{'='*70}\n  OVERNIGHT vs INTRADAY — where do the returns actually happen?\n{'='*70}")
    print(f"  {'Ticker':<7}{'Overnight':>12}{'Intraday':>12}{'Buy&Hold':>12}  (5yr compounded)")
    print(f"  {'-'*55}")
    on_all=[]; id_all=[]
    agg_on=None; agg_id=None; agg_full=None
    for sym in NAMES:
        d = load(sym)
        if d is None: continue
        on, idr, fu = compound(d.overnight), compound(d.intraday), compound(d.full)
        print(f"  {sym:<7}{on:>+11.0f}%{idr:>+11.0f}%{fu:>+11.0f}%")
        on_all.append(d.overnight.mean()); id_all.append(d.intraday.mean())

    print(f"  {'-'*55}")
    print(f"\n  Average DAILY return decomposition across {len(on_all)} names:")
    print(f"    Overnight: {np.mean(on_all):+.4f}%/day   (annualized ~{np.mean(on_all)*252:+.1f}%)")
    print(f"    Intraday : {np.mean(id_all):+.4f}%/day   (annualized ~{np.mean(id_all)*252:+.1f}%)")

    # Tradeability reality check (SPY, daily round-trip)
    spy = load("SPY")
    if spy is not None:
        n = len(spy)
        gross = spy.overnight.mean()
        print(f"\n  TRADEABILITY (overnight-only on SPY, {n} days):")
        print(f"    Gross overnight edge: {gross:+.4f}%/day")
        for cost in (0.01, 0.02, 0.05):
            net = gross - cost
            print(f"    After {cost:.2f}%/day round-trip cost: {net:+.4f}%/day  (ann ~{net*252:+.1f}%)")
        print("    (You'd buy market-on-close, sell market-on-open EVERY day.)")
    print(f"{'='*70}\n")
