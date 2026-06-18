#!/usr/bin/env python3
"""Evidence-based optimization sweep across the watchlist (1yr)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
import backtest as bt

WATCHLIST = ["ARM","AMD","GOOGL","COIN","NFLX","AAPL","AMZN","MSFT","QQQ","MSTR","HOOD",
             "TSLA","PLTR","NVDA","META","SOFI","MU","AVGO","SMCI","SPY"]
EVAL_DAYS = 252

spy = yf.Ticker("SPY").history(period="2y")[["Open","High","Low","Close","Volume"]]
spy.index = spy.index.tz_localize(None)

# Collect every signal with its score + forward outcomes at several configs
records = []
for sym in WATCHLIST:
    try:
        d = yf.Ticker(sym).history(period="2y")[["Open","High","Low","Close","Volume"]]
        d.index = d.index.tz_localize(None)
        common = d.index.intersection(spy.index); d, s = d.loc[common], spy.loc[common]
        if len(d) < 260: continue
        for i in range(len(d)-EVAL_DAYS, len(d)):
            if i < 220 or i+1 >= len(d): continue
            w, sw = d.iloc[:i+1], s.iloc[:i+1]
            direction, score, adx, rsi = bt.score_bar(w, sw)
            if direction == "NEUTRAL" or score < 5: continue
            entry = float(d.Open.iloc[i+1])
            # forward returns at multiple horizons
            fwd = {}
            for hb in (1,3,5,10):
                k = min(i+hb, len(d)-1)
                px = float(d.Close.iloc[k])
                fwd[hb] = (px-entry)/entry*100 * (1 if direction=="BULLISH" else -1)
            records.append({"sym":sym, "score":score, "adx":adx, "rsi":rsi, **{f"r{h}":fwd[h] for h in (1,3,5,10)}})
    except Exception: continue

df = pd.DataFrame(records)
print(f"\n{'='*64}\n  OPTIMIZATION SWEEP — {len(df)} signals, 1 year, full universe\n{'='*64}")

# 1) Does score predict outcome? (3-day horizon)
print("\n  ① SCORE vs 3-DAY RETURN")
print(f"  {'Score':>6}{'N':>6}{'Win%':>7}{'AvgRet%':>9}")
for sc in sorted(df.score.unique()):
    sub = df[df.score==sc]
    if len(sub) < 5: continue
    print(f"  {sc:>6}{len(sub):>6}{(sub.r3>0).mean()*100:>6.0f}%{sub.r3.mean():>+8.2f}%")

# 2) Best hold period (all signals score>=8)
print("\n  ② HOLD PERIOD (signals score≥8)")
strong = df[df.score>=8]
print(f"  {'Hold':>6}{'Win%':>7}{'AvgRet%':>9}")
for h in (1,3,5,10):
    print(f"  {h:>5}d{(strong[f'r{h}']>0).mean()*100:>6.0f}%{strong[f'r{h}'].mean():>+8.2f}%")

# 3) ADX bucket (score>=8, 5-day)
print("\n  ③ ADX STRENGTH (score≥8, 5-day return)")
for lo,hi in [(0,20),(20,30),(30,45),(45,100)]:
    sub = strong[(strong.adx>=lo)&(strong.adx<hi)]
    if len(sub)<5: continue
    print(f"  ADX {lo:>2}-{hi:<3}{len(sub):>5}  win {(sub.r5>0).mean()*100:>3.0f}%  avg {sub.r5.mean():>+5.2f}%")

# 4) RSI bucket (score>=8, 5-day)
print("\n  ④ RSI ZONE (score≥8, 5-day return)")
for lo,hi in [(0,40),(40,50),(50,60),(60,70),(70,100)]:
    sub = strong[(strong.rsi>=lo)&(strong.rsi<hi)]
    if len(sub)<5: continue
    print(f"  RSI {lo:>2}-{hi:<3}{len(sub):>5}  win {(sub.r5>0).mean()*100:>3.0f}%  avg {sub.r5.mean():>+5.2f}%")

print(f"\n{'='*64}\n")
