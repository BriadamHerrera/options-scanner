#!/usr/bin/env python3
"""
"Just invert the losing strategy" — the most common trading myth. Tested.

If SMC loses because it has a strong NEGATIVE edge, inverting it should win.
If SMC loses because it has ~ZERO edge and COSTS kill it, inverting it just
gives another ~zero-edge strategy that costs ALSO kill (you can't flip costs).

We run original AND inverted signals, measured on the UNDERLYING (no costs)
and with OPTION costs, so the mechanism is unmistakable.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import smc_test as smc

def backtest(signal_fn, invert=False):
    u_all=[]; o_all=[]
    for sym in smc.NAMES:
        df=yf.Ticker(sym).history(period="5y")[["Open","High","Low","Close","Volume"]]
        if len(df)<800: continue
        df.index=df.index.tz_localize(None)
        sig=signal_fn(df)
        for i in range(260, len(df)-1):
            direction=sig.iloc[i]
            if direction not in ("BULLISH","BEARISH"): continue
            bull = (direction=="BULLISH")
            if invert: bull = not bull
            u,held=smc.simulate(df,i,bull)
            u_all.append(u); o_all.append(smc.option_ret(u,held))
    return np.array(u_all), np.array(o_all)

def line(label, u, o):
    print(f"  {label:<34}{len(u):>6}{u.mean():>+9.3f}%{o.mean():>+10.2f}%")

if __name__=="__main__":
    print(f"\n{'='*68}\n  CAN YOU JUST INVERT A LOSING STRATEGY?\n{'='*68}")
    print(f"  {'Strategy':<34}{'N':>6}{'Underlying':>10}{'Option':>10}")
    print(f"  {'-'*54}")
    for name, fn in [("Liquidity Sweep", smc.liquidity_sweep_signals),
                     ("Fair Value Gap",  smc.fvg_signals)]:
        u,o   = backtest(fn, invert=False)
        ui,oi = backtest(fn, invert=True)
        line(f"{name} (original)", u, o)
        line(f"{name} (INVERTED)", ui, oi)
        print(f"  {'-'*54}")
    print("\n  Read: if 'Underlying' is ~0 for both directions, the signal has NO")
    print("  directional edge to invert — and option costs sink BOTH sides.")
    print(f"{'='*68}\n")
