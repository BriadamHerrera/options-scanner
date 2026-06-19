#!/usr/bin/env python3
"""
Rigorous validation of the Supertrend reversal signal:
  • Multiple rolling out-of-sample windows (not one lucky year)
  • Realistic OPTION cost model layered on the underlying move

Cost model (per round-trip, applied to the option, not the stock):
  • Theta: long ATM ~45 DTE option loses ~0.12%/day of its value to time decay.
    We hold `held` days, so theta_cost ≈ 0.12% * held * option_leverage_offset.
    Simpler & honest: convert underlying move → option move via delta leverage,
    then subtract theta + spread in OPTION terms.

Approximation used (transparent, not magic):
  ATM ~45DTE option ≈ 5x leverage on the underlying move (delta~0.5, premium~10%
  of notional → 1% stock ≈ ~5% option). Theta ≈ 0.7%/day of premium. Spread ≈ 4%
  round trip. So:  option_ret% ≈ 5 * underlying_ret% − 0.7*held − 4
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import chartprime_reversal as cp

cp.ATR_PERIOD, cp.ATR_MULT = 10, 2.0   # the robust setting from the screen

NAMES = cp.ALL_NAMES
LEVERAGE   = 5.0     # ATM ~45DTE option vs underlying
THETA_DAY  = 0.7     # % of premium lost per calendar day
SPREAD_RT  = 4.0     # % round-trip bid/ask cost

def windows(n, span=252, step=126, count=6):
    """Return list of (start,end) index ranges walking backward from the end."""
    out=[]
    end = n
    for _ in range(count):
        start = end - span
        if start < 300: break
        out.append((start, end))
        end -= step
    return out[::-1]

def collect_trades(names):
    """Run supertrend over full history; return list of (entry_idx_global, held, underlying_ret) per symbol window-agnostic."""
    data={}
    for sym in names:
        df = yf.Ticker(sym).history(period="5y")[["Open","High","Low","Close","Volume"]]
        if len(df) < 800: continue
        df.index = df.index.tz_localize(None)
        sig, d = cp.signals(df)
        trades=[]  # (i, held, uret)
        for i in range(260, len(df)-1):
            direction = sig.iloc[i]
            if direction not in ("BULLISH","BEARISH"): continue
            entry=float(df.Open.iloc[i+1]); bull=direction=="BULLISH"; ex=None; held=0
            for j in range(i+1, min(i+1+cp.HOLD_BARS, len(df))):
                held=j-i
                hi,lo=float(df.High.iloc[j]),float(df.Low.iloc[j])
                if (bull and d.iloc[j]==-1) or (not bull and d.iloc[j]==1):
                    ex=float(df.Close.iloc[j]); break
                if bull and lo<=entry*(1-cp.STOP_PCT/100): ex=entry*(1-cp.STOP_PCT/100); break
                if not bull and hi>=entry*(1+cp.STOP_PCT/100): ex=entry*(1+cp.STOP_PCT/100); break
            if ex is None: ex=float(df.Close.iloc[min(i+cp.HOLD_BARS,len(df)-1)]); held=cp.HOLD_BARS
            uret=(ex-entry)/entry*100*(1 if bull else -1)
            trades.append((i, max(held,1), uret))
        data[sym]=(len(df), trades)
    return data

def option_ret(uret, held):
    return LEVERAGE*uret - THETA_DAY*held - SPREAD_RT

if __name__ == "__main__":
    print(f"\n{'='*72}\n  SUPERTREND VALIDATION — multi-window + option costs\n{'='*72}")
    print(f"  Signal: Supertrend({cp.ATR_PERIOD},{cp.ATR_MULT}) reversal, exit on opposite flip")
    print(f"  Option model: {LEVERAGE}x leverage − {THETA_DAY}%/day theta − {SPREAD_RT}% spread\n")

    data = collect_trades(NAMES)
    # Use one ticker's length as reference for window bounds (they share calendar approx)
    ref_n = max(v[0] for v in data.values())
    wins = windows(ref_n)

    print(f"  {'Window (bars back)':<22}{'N':>5}{'Win%':>7}{'Underlying':>12}{'OPTION exp':>12}{'OPT total':>11}")
    print(f"  {'-'*68}")
    grand_opt=[]
    for (s,e) in wins:
        u_all=[]; o_all=[]
        for sym,(n,trades) in data.items():
            off = n - ref_n  # align each symbol's index to ref window
            for (i,held,uret) in trades:
                gi = i  # symbol-local index
                # map window in ref space to this symbol: same calendar position
                if s+off <= gi < e+off:
                    u_all.append(uret); o_all.append(option_ret(uret,held))
        if not o_all: continue
        u=np.array(u_all); o=np.array(o_all)
        grand_opt += o_all
        label=f"{ref_n-e}-{ref_n-s} ago"
        print(f"  {label:<22}{len(o):>5}{np.mean(o>0)*100:>6.0f}%{u.mean():>+11.2f}%{o.mean():>+11.2f}%{o.sum():>+10.1f}%")

    print(f"  {'-'*68}")
    if grand_opt:
        g=np.array(grand_opt)
        print(f"  {'ALL WINDOWS (options)':<22}{len(g):>5}{np.mean(g>0)*100:>6.0f}%{'':>12}{g.mean():>+11.2f}%{g.sum():>+10.1f}%")
        pos = sum(1 for (s,e) in wins if True)
        print(f"\n  Verdict: option expectancy {'POSITIVE' if g.mean()>0 else 'NEGATIVE'} after costs "
              f"({g.mean():+.2f}%/trade across {len(g)} trades).")
    print(f"{'='*72}\n")
