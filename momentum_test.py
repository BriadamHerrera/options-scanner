#!/usr/bin/env python3
"""
Hunting a REAL edge: cross-sectional momentum (relative strength).

Hypothesis (with mechanism): investors underreact to information and then herd,
so stocks that outperformed over the past ~12 months keep outperforming for a
few months. The most documented anomaly in finance (Jegadeesh-Titman 1993; works
100+ yrs, across asset classes).

Method:
  • Diversified multi-sector universe (NOT the biased high-vol tech list)
  • Each month: rank by trailing 12-1 month return (skip last month = avoid
    short-term reversal). Long the top quintile, equal weight. Rebalance monthly.
  • Compare vs equal-weight universe AND vs SPY. Apply turnover cost.
  • Stock returns (this is a portfolio strategy, not options).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

UNIVERSE = [
    # Tech
    "AAPL","MSFT","NVDA","GOOGL","META","AMD","AVGO","CRM","ORCL","ADBE",
    # Financials
    "JPM","BAC","GS","MS","V","MA","WFC","AXP",
    # Healthcare
    "JNJ","UNH","PFE","MRK","ABBV","LLY","TMO",
    # Consumer
    "AMZN","WMT","HD","MCD","NKE","COST","PG","KO","PEP","SBUX",
    # Energy / Industrials
    "XOM","CVX","COP","CAT","BA","HON","GE","UPS",
    # Other
    "DIS","NFLX","TSLA","VZ",
]
TOP_FRAC = 0.20          # long the top quintile
COST_BPS = 0.10          # per-name rebalance cost (%)

def load_monthly():
    px = yf.download(UNIVERSE + ["SPY"], period="6y", interval="1mo",
                     auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    return px

def backtest():
    px = load_monthly()
    spy = px["SPY"]
    uni = px[[c for c in UNIVERSE if c in px.columns]].copy()
    months = uni.index

    port_rets, ew_rets, spy_rets, dates = [], [], [], []
    prev_holds = set()
    for t in range(12, len(months)-1):
        # 12-1 momentum: return from t-12 to t-1 (skip the most recent month)
        mom = uni.iloc[t-1] / uni.iloc[t-12] - 1
        mom = mom.dropna()
        if len(mom) < 10: continue
        n = max(1, int(len(mom)*TOP_FRAC))
        winners = list(mom.sort_values(ascending=False).head(n).index)

        # next-month return of the held basket
        nxt = uni.iloc[t+1] / uni.iloc[t] - 1
        port = nxt[winners].mean()
        # turnover cost: names changed since last month
        turnover = len(set(winners) ^ prev_holds)
        cost = turnover * COST_BPS/100 / max(len(winners),1)
        port -= cost
        prev_holds = set(winners)

        port_rets.append(port)
        ew_rets.append(nxt.mean())
        spy_rets.append(spy.iloc[t+1]/spy.iloc[t]-1)
        dates.append(months[t+1])

    df = pd.DataFrame({"momentum":port_rets, "equal_weight":ew_rets, "spy":spy_rets}, index=dates)
    return df

def stats(label, r):
    r = pd.Series(r).dropna()
    ann = (1+r).prod()**(12/len(r))-1
    vol = r.std()*np.sqrt(12)
    sharpe = ann/vol if vol>0 else 0
    total = (1+r).prod()-1
    mdd = ((1+r).cumprod()/(1+r).cumprod().cummax()-1).min()
    print(f"  {label:<16} ann {ann*100:>+6.1f}%  vol {vol*100:>4.0f}%  Sharpe {sharpe:>4.2f}  total {total*100:>+7.0f}%  maxDD {mdd*100:>5.0f}%")

if __name__=="__main__":
    print(f"\n{'='*72}\n  CROSS-SECTIONAL MOMENTUM — relative strength, monthly rebalance\n{'='*72}")
    df = backtest()
    print(f"  Months tested: {len(df)}  | Universe: {len(UNIVERSE)} names, multi-sector\n")
    stats("Momentum (top 20%)", df["momentum"])
    stats("Equal-weight uni", df["equal_weight"])
    stats("SPY (buy & hold)", df["spy"])
    # year-by-year momentum vs spy
    print(f"\n  Year-by-year (momentum total return vs SPY):")
    by = df.groupby(df.index.year)
    for yr, g in by:
        m = (1+g["momentum"]).prod()-1; s = (1+g["spy"]).prod()-1
        flag = "✅" if m>s else "  "
        print(f"    {yr}:  momentum {m*100:>+6.1f}%   spy {s*100:>+6.1f}%  {flag}")
    print(f"{'='*72}\n")
