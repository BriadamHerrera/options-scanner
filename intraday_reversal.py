#!/usr/bin/env python3
"""
Intraday Trend-Reversal strategy + backtest (5min / 15min bars).

Catches intraday exhaustion turns: price stretches far from VWAP, RSI hits an
extreme and starts to turn, an exhaustion candle prints on climax volume → fade
the move back toward VWAP.

⚠️ DATA LIMIT: free intraday history is only ~60 days, so this CANNOT be
out-of-sample validated like the daily strategies. Treat results as a tiny,
in-sample sample — paper-trade only.

Scoring (0–9):
  1. VWAP deviation extreme (>1.5%)        → +2
  2. RSI(14) extreme & turning             → +2
  3. Exhaustion candle at the extreme      → +2
  4. Volume climax (>2x avg)               → +1
  5. Bollinger-band pierce & re-enter      → +1
  6. Real prior move to reverse (>1%)      → +1

Exit: target = revert to VWAP | stop = -1.2% | time = 8 bars | EOD close.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

SYMBOL    = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
INTERVAL  = sys.argv[2] if len(sys.argv) > 2 else "15m"
MIN_SCORE = 5
HOLD_BARS = 8
STOP_PCT  = 1.2

def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def rsi(close, period=14):
    d=close.diff()
    g=d.clip(lower=0).ewm(span=period,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(span=period,adjust=False).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def add_vwap(df):
    """Intraday VWAP, reset each session."""
    tp = (df.High+df.Low+df.Close)/3
    df = df.copy()
    df["date"] = df.index.date
    df["cum_tpv"] = (tp*df.Volume).groupby(df["date"]).cumsum()
    df["cum_v"]   = df.Volume.groupby(df["date"]).cumsum()
    df["vwap"]    = df["cum_tpv"]/df["cum_v"].replace(0,np.nan)
    return df

def rev_score(w):
    """Reversal score at last bar of window w (must have vwap, rsi columns)."""
    c=float(w.Close.iloc[-1]); o=float(w.Open.iloc[-1]); h=float(w.High.iloc[-1]); l=float(w.Low.iloc[-1])
    vwap=float(w.vwap.iloc[-1])
    if vwap<=0 or np.isnan(vwap): return "NEUTRAL",0
    dev=(c-vwap)/vwap*100
    r=float(w.rsi.iloc[-1]); r_prev=float(w.rsi.iloc[-2])
    rng=h-l
    if rng==0: return "NEUTRAL",0
    body=abs(c-o)/rng; close_pos=(c-l)/rng

    # direction: stretched BELOW vwap & oversold turning up = BULLISH reversal
    oversold   = dev < -1.5 and r < 35
    overbought = dev >  1.5 and r > 65
    if oversold:    direction="BULLISH"
    elif overbought: direction="BEARISH"
    else: return "NEUTRAL",0

    score=0
    if abs(dev) >= 1.5: score+=2
    # RSI extreme AND turning back
    if direction=="BULLISH" and r<35 and r>r_prev: score+=2
    if direction=="BEARISH" and r>65 and r<r_prev: score+=2
    # exhaustion candle: bullish hammer (close upper half) / bearish star (close lower half)
    if direction=="BULLISH" and close_pos>=0.6 and body<0.6: score+=2
    if direction=="BEARISH" and close_pos<=0.4 and body<0.6: score+=2
    # volume climax
    vavg=float(w.Volume.iloc[-21:-1].mean()); vr=float(w.Volume.iloc[-1])/vavg if vavg>0 else 1
    if vr>=2.0: score+=1
    # bollinger pierce & re-enter
    sma20=float(w.Close.rolling(20).mean().iloc[-1]); std20=float(w.Close.rolling(20).std().iloc[-1])
    if std20>0:
        if direction=="BULLISH" and float(w.Low.iloc[-1])<sma20-2*std20 and c>sma20-2*std20: score+=1
        if direction=="BEARISH" and float(w.High.iloc[-1])>sma20+2*std20 and c<sma20+2*std20: score+=1
    # prior move magnitude (last 6 bars)
    mv=abs(c-float(w.Close.iloc[-6]))/float(w.Close.iloc[-6])*100 if len(w)>=6 else 0
    if mv>=1.0: score+=1
    return direction, score


if __name__ == "__main__":
    print(f"\n{'='*64}\n  INTRADAY REVERSAL: {SYMBOL} {INTERVAL} (~60d, in-sample only)\n{'='*64}")
    df = yf.Ticker(SYMBOL).history(period="60d", interval=INTERVAL)
    if df.empty or len(df) < 60:
        print("  Not enough intraday data."); sys.exit()
    df = df[["Open","High","Low","Close","Volume"]]
    df = add_vwap(df)
    df["rsi"] = rsi(df.Close)
    trades=[]
    for i in range(30, len(df)-1):
        w=df.iloc[:i+1]
        # only trade if same session continues for entry
        direction, score = rev_score(w)
        if direction=="NEUTRAL" or score<MIN_SCORE: continue
        entry=float(df.Open.iloc[i+1]); bull=direction=="BULLISH"
        sess=df["date"].iloc[i]; ex=None
        for j in range(i+1, min(i+1+HOLD_BARS, len(df))):
            if df["date"].iloc[j]!=sess: ex=float(df.Close.iloc[j-1]); break  # close at EOD
            hi,lo,cl=float(df.High.iloc[j]),float(df.Low.iloc[j]),float(df.Close.iloc[j])
            vw=float(df.vwap.iloc[j])
            if bull:
                if lo<=entry*(1-STOP_PCT/100): ex=entry*(1-STOP_PCT/100); break
                if cl>=vw: ex=cl; break  # reverted to vwap = target
            else:
                if hi>=entry*(1+STOP_PCT/100): ex=entry*(1+STOP_PCT/100); break
                if cl<=vw: ex=cl; break
        if ex is None: ex=float(df.Close.iloc[min(i+HOLD_BARS,len(df)-1)])
        trades.append((ex-entry)/entry*100*(1 if bull else -1))
    print(f"{'-'*64}")
    if trades:
        a=np.array(trades); w=a[a>0]
        print(f"  Signals {len(a)} | Win {len(w)/len(a)*100:.0f}% | Avg {a.mean():+.2f}% | Total {a.sum():+.1f}% (underlying)")
        print(f"  ⚠️ ~60-day sample, in-sample only — NOT statistically reliable.")
    else:
        print("  No signals.")
    print(f"{'='*64}\n")
