#!/usr/bin/env python3
"""
Forward paper-trader for the optimized Anti-Fakeout swing strategy.

Runs ONCE PER DAY (after market close). Each run:
  1. Manages open positions — checks stop / trailing-stop / target / time exits
     using the day's high/low, and records the REAL option P/L (theta included).
  2. Scans the robust watchlist for fresh signals (score ≥ 9) and opens new
     paper positions at the current option mid price.

Everything is logged to:
  • paper_state.json  — currently open positions
  • paper_trades.csv  — closed-trade ledger (the forward-test record)

This is the honest validation the backtest couldn't give you: real option
entry/exit prices, real theta decay, real bid/ask. After 1–2 months the CSV
tells you whether the strategy actually makes money on contracts, not just
on the underlying.

Usage:  python3 paper_trader.py
"""

import os, json, csv, warnings
warnings.filterwarnings("ignore")
from datetime import datetime, date

import numpy as np
import pandas as pd
import yfinance as yf

import backtest as bt  # reuse the exact scoring engine

# ─── CONFIG (mirrors optimized strategy) ─────────────────────────────────────────
WATCHLIST  = ["AMD","GOOGL","ARM","HOOD","QQQ","MSTR","AMZN","COIN","SPY"]
MIN_SCORE  = 9
HOLD_DAYS  = 10
STOP_PCT   = 4.0
TRAIL_PCT  = 4.0
TRAIL_ARM  = 4.0
TGT_PCT    = 12.0
TARGET_DTE = 45
MIN_OI     = 100

HERE        = os.path.dirname(os.path.abspath(__file__))
STATE_FILE  = os.path.join(HERE, "paper_state.json")
TRADES_FILE = os.path.join(HERE, "paper_trades.csv")

TRADE_COLS = ["entry_date","exit_date","symbol","type","strike","expiry","direction",
              "score","entry_underlying","exit_underlying","underlying_ret%",
              "entry_mid","exit_mid","option_ret%","outcome","days_held"]


# ─── STATE I/O ───────────────────────────────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {"positions": []}

def save_state(state):
    with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2)

def log_trade(row):
    new = not os.path.exists(TRADES_FILE)
    with open(TRADES_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_COLS)
        if new: w.writeheader()
        w.writerow(row)


# ─── OPTION HELPERS ──────────────────────────────────────────────────────────────
def pick_contract(tk, direction, spot):
    """Find the near-the-money contract closest to TARGET_DTE. Returns dict or None."""
    try:
        exps = tk.options
        if not exps: return None
        today = date.today()
        best, bestdiff = None, 999
        for e in exps:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            if dte < 7: continue
            if abs(dte - TARGET_DTE) < bestdiff: bestdiff, best = abs(dte - TARGET_DTE), e
        if not best: return None
        chain = tk.option_chain(best)
        df = chain.calls if direction == "BULLISH" else chain.puts
        df = df[df["openInterest"] >= MIN_OI].copy()
        if df.empty: return None
        df["dist"] = (df["strike"] - spot).abs()
        pool = df[df["strike"] >= spot*0.99] if direction=="BULLISH" else df[df["strike"] <= spot*1.01]
        row = (pool if not pool.empty else df).sort_values("dist").iloc[0]
        return {"strike": float(row["strike"]), "expiry": best,
                "type": "CALL" if direction=="BULLISH" else "PUT",
                "mid": option_mid(row)}
    except Exception:
        return None

def option_mid(row):
    bid = float(row.get("bid",0) or 0); ask = float(row.get("ask",0) or 0)
    if bid and ask: return round((bid+ask)/2, 2)
    return round(float(row.get("lastPrice",0) or 0), 2)

def current_option_mid(symbol, expiry, strike, opt_type):
    """Re-price an existing contract at today's mid."""
    try:
        tk = yf.Ticker(symbol)
        chain = tk.option_chain(expiry)
        df = chain.calls if opt_type == "CALL" else chain.puts
        r = df[df["strike"] == strike]
        if r.empty: return None
        return option_mid(r.iloc[0])
    except Exception:
        return None


# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    today = date.today()
    print(f"\n{'='*66}\n  PAPER TRADER — {today}  (optimized swing strategy)\n{'='*66}")

    state = load_state()
    held_syms = {p["symbol"] for p in state["positions"]}

    spy = yf.Ticker("SPY").history(period="2y")[["Open","High","Low","Close","Volume"]]
    spy.index = spy.index.tz_localize(None)

    # ── 1. MANAGE OPEN POSITIONS ─────────────────────────────────────────────────
    still_open = []
    for p in state["positions"]:
        sym = p["symbol"]
        d = yf.Ticker(sym).history(period="1mo")[["Open","High","Low","Close"]]
        if d.empty:
            still_open.append(p); continue
        d.index = d.index.tz_localize(None)
        hi, lo, close = float(d["High"].iloc[-1]), float(d["Low"].iloc[-1]), float(d["Close"].iloc[-1])
        entry = p["entry_underlying"]; bull = p["direction"] == "BULLISH"

        # update peak favorable price
        peak = p["peak_underlying"]
        peak = max(peak, hi) if bull else min(peak, lo)
        p["peak_underlying"] = peak

        days = int(np.busday_count(p["entry_date"], str(today)))
        exit_now, outcome, exit_u = False, None, close

        if bull:
            if lo <= entry*(1-STOP_PCT/100):                       exit_now,outcome,exit_u = True,"STOP",   entry*(1-STOP_PCT/100)
            elif peak >= entry*(1+TRAIL_ARM/100) and lo <= peak*(1-TRAIL_PCT/100): exit_now,outcome,exit_u = True,"TRAIL", peak*(1-TRAIL_PCT/100)
            elif hi >= entry*(1+TGT_PCT/100):                      exit_now,outcome,exit_u = True,"TARGET", entry*(1+TGT_PCT/100)
        else:
            if hi >= entry*(1+STOP_PCT/100):                       exit_now,outcome,exit_u = True,"STOP",   entry*(1+STOP_PCT/100)
            elif peak <= entry*(1-TRAIL_ARM/100) and hi >= peak*(1+TRAIL_PCT/100): exit_now,outcome,exit_u = True,"TRAIL", peak*(1+TRAIL_PCT/100)
            elif lo <= entry*(1-TGT_PCT/100):                      exit_now,outcome,exit_u = True,"TARGET", entry*(1-TGT_PCT/100)
        if not exit_now and days >= HOLD_DAYS:
            exit_now, outcome, exit_u = True, "TIME", close

        if exit_now:
            exit_mid = current_option_mid(sym, p["expiry"], p["strike"], p["type"]) or p["entry_mid"]
            opt_ret  = (exit_mid - p["entry_mid"]) / p["entry_mid"] * 100
            u_ret    = (exit_u - entry)/entry*100 * (1 if bull else -1)
            log_trade({
                "entry_date":p["entry_date"], "exit_date":str(today), "symbol":sym,
                "type":p["type"], "strike":p["strike"], "expiry":p["expiry"],
                "direction":p["direction"], "score":p["score"],
                "entry_underlying":round(entry,2), "exit_underlying":round(exit_u,2),
                "underlying_ret%":round(u_ret,2), "entry_mid":p["entry_mid"],
                "exit_mid":exit_mid, "option_ret%":round(opt_ret,1),
                "outcome":outcome, "days_held":days,
            })
            print(f"  CLOSE {sym} {p['type']} ${p['strike']} — {outcome} | "
                  f"underlying {u_ret:+.1f}% | OPTION {opt_ret:+.1f}% ({days}d)")
        else:
            still_open.append(p)
    state["positions"] = still_open
    held_syms = {p["symbol"] for p in still_open}

    # ── 2. SCAN FOR NEW SIGNALS ──────────────────────────────────────────────────
    print(f"\n  Scanning {len(WATCHLIST)} names for new signals (score ≥{MIN_SCORE})…")
    for sym in WATCHLIST:
        if sym in held_syms:
            continue  # one position per name at a time
        try:
            d = yf.Ticker(sym).history(period="2y")[["Open","High","Low","Close","Volume"]]
            d.index = d.index.tz_localize(None)
            common = d.index.intersection(spy.index); d2, s2 = d.loc[common], spy.loc[common]
            if len(d2) < 220: continue
            direction, score, adx, rsi = bt.score_bar(d2, s2)
            if direction == "NEUTRAL" or score < MIN_SCORE:
                continue
            tk = yf.Ticker(sym)
            spot = float(d2["Close"].iloc[-1])
            c = pick_contract(tk, direction, spot)
            if not c or c["mid"] <= 0:
                print(f"  {sym}: signal score={score} but no tradeable contract — skipped")
                continue
            pos = {"symbol":sym, "direction":direction, "entry_date":str(today),
                   "entry_underlying":round(spot,2), "peak_underlying":round(spot,2),
                   "strike":c["strike"], "expiry":c["expiry"], "type":c["type"],
                   "entry_mid":c["mid"], "score":score}
            state["positions"].append(pos)
            print(f"  OPEN  {sym} {c['type']} ${c['strike']} exp {c['expiry']} "
                  f"@ ${c['mid']:.2f} | score {score}/13 adx={adx} rsi={rsi}")
        except Exception as e:
            continue

    save_state(state)

    # ── 3. SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\n  Open positions: {len(state['positions'])}")
    if os.path.exists(TRADES_FILE):
        df = pd.read_csv(TRADES_FILE)
        if len(df):
            wr = (df["option_ret%"] > 0).mean()*100
            print(f"  Closed trades : {len(df)}  |  win rate {wr:.0f}%  |  "
                  f"avg option {df['option_ret%'].mean():+.1f}%  |  "
                  f"total {df['option_ret%'].sum():+.1f}%")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
