#!/usr/bin/env python3
"""Portfolio backtest — runs the scanner signal across the whole watchlist."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

# reuse the scoring engine from backtest.py
import backtest as bt

WATCHLIST = ["SPY","QQQ","AAPL","TSLA","NVDA","AMD","MSFT","AMZN","META","GOOGL",
             "NFLX","COIN","MSTR","PLTR","ARM","SMCI","AVGO","MU","SOFI","HOOD"]
EVAL_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 126  # ~6 months

spy = yf.Ticker("SPY").history(period="2y")[["Open","High","Low","Close","Volume"]]
spy.index = spy.index.tz_localize(None)

all_trades, per_ticker = [], {}
for sym in WATCHLIST:
    try:
        data = yf.Ticker(sym).history(period="2y")[["Open","High","Low","Close","Volume"]]
        data.index = data.index.tz_localize(None)
        common = data.index.intersection(spy.index)
        d, s = data.loc[common], spy.loc[common]
        if len(d) < 260: continue
        start = len(d) - EVAL_DAYS
        tr = []
        for i in range(start, len(d)):
            w, sw = d.iloc[:i+1], s.iloc[:i+1]
            if len(w) < 220 or i+1 >= len(d): continue
            direction, score, _, _ = bt.score_bar(w, sw)
            if direction == "NEUTRAL" or score < bt.MIN_SCORE: continue
            entry = float(d.Open.iloc[i+1]); exit_px = None
            for j in range(i+1, min(i+1+bt.HOLD_BARS, len(d))):
                hi, lo = float(d.High.iloc[j]), float(d.Low.iloc[j])
                if direction == "BULLISH":
                    if lo <= entry*(1-bt.STOP_PCT/100): exit_px = entry*(1-bt.STOP_PCT/100); break
                    if hi >= entry*(1+bt.TGT_PCT/100):  exit_px = entry*(1+bt.TGT_PCT/100); break
                else:
                    if hi >= entry*(1+bt.STOP_PCT/100): exit_px = entry*(1+bt.STOP_PCT/100); break
                    if lo <= entry*(1-bt.TGT_PCT/100):  exit_px = entry*(1-bt.TGT_PCT/100); break
            if exit_px is None: exit_px = float(d.Close.iloc[min(i+bt.HOLD_BARS, len(d)-1)])
            ret = (exit_px-entry)/entry*100 * (1 if direction=="BULLISH" else -1)
            tr.append(ret); all_trades.append(ret)
        if tr:
            per_ticker[sym] = (len(tr), sum(1 for x in tr if x>0)/len(tr)*100, np.mean(tr), sum(tr))
    except Exception as e:
        continue

print(f"\n{'='*70}\n  PORTFOLIO BACKTEST — {len(WATCHLIST)} tickers, last {EVAL_DAYS} trading days\n{'='*70}")
print(f"  {'Ticker':<8}{'Trades':>7}{'Win%':>7}{'Avg%':>9}{'Total%':>10}")
print(f"  {'-'*40}")
for sym, (n, wr, avg, tot) in sorted(per_ticker.items(), key=lambda x: -x[1][3]):
    print(f"  {sym:<8}{n:>7}{wr:>6.0f}%{avg:>+8.2f}%{tot:>+9.1f}%")

print(f"  {'-'*40}")
if all_trades:
    wins = [t for t in all_trades if t > 0]
    print(f"\n  TOTAL signals : {len(all_trades)}")
    print(f"  Win rate      : {len(wins)/len(all_trades)*100:.0f}%")
    print(f"  Avg/trade     : {np.mean(all_trades):+.2f}% (underlying)")
    print(f"  Sum of returns: {sum(all_trades):+.1f}% (underlying)")
    print(f"  Expectancy    : {np.mean(all_trades):+.2f}% per trade")
print(f"{'='*70}\n")
