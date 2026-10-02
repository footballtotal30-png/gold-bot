"""
XAUUSD (Gold) A+ setup bot -> Telegram. Runs hourly on GitHub Actions (free).
A+ = 5/5 score on H1 + fresh signal + H4 trend agrees + ADX > 20
     + price not overextended + London/New York hours.
Not guaranteed. Test on a demo account first.
"""
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TICKERS = ["XAUUSD=X", "GC=F"]   # spot gold first, futures as backup
OFFSET = 0.0       # broker price minus Yahoo price (e.g. 2.5). Compare once in MT5.
MIN_SCORE = 5      # 5 = strictest. Use 4 for more signals.
SESSION = (7, 20)  # UTC hours
TOKEN, CHAT = os.getenv("TELEGRAM_TOKEN"), os.getenv("CHAT_ID")


def send(text):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  data={"chat_id": CHAT, "text": text}, timeout=20)


def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def load():
    for t in TICKERS:
        df = yf.download(t, period="120d", interval="60m", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna().rename(columns=str.lower)
        if len(df) > 300:
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC")
            return df.iloc[:-1]   # drop the candle still forming
    return None


def analyse():
    df = load()
    if df is None:
        return None, "No gold data from Yahoo right now."
    c, h, l = df["close"], df["high"], df["low"]
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    e50, e200 = ema(c, 50), ema(c, 200)
    hh, ll = h.rolling(20).max().shift(), l.rolling(20).min().shift()

    pdm = h.diff().where((h.diff() > -l.diff()) & (h.diff() > 0), 0.0)
    mdm = (-l.diff()).where((-l.diff() > h.diff()) & (-l.diff() > 0), 0.0)
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1/14, adjust=False).mean() / atr
    adx = (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(alpha=1/14, adjust=False).mean()

    score = (np.sign(e50 - e200) + np.sign(c - e50) + np.sign(hist)
             + ((rsi > 50) & (rsi < 70)).astype(int) - ((rsi < 50) & (rsi > 30)).astype(int)
             + (c > hh).astype(int) - (c < ll).astype(int))
    h4 = c.resample("4h").last().dropna()
    h4_up = ema(h4, 50).iloc[-1] > ema(h4, 200).iloc[-1]

    s, sp = score.iloc[-1], score.iloc[-2]
    px, a = c.iloc[-1] + OFFSET, atr.iloc[-1]
    status = (f"Gold {px:.0f} | score {s:+.0f}/5 | ADX {adx.iloc[-1]:.0f} | "
              f"H4 {'up' if h4_up else 'down'}")

    last = df.index[-1]
    ok = (SESSION[0] <= last.hour < SESSION[1]
          and not (last.weekday() == 4 and last.hour >= 18)
          and adx.iloc[-1] > 20
          and abs(c.iloc[-1] - e50.iloc[-1]) < 2.5 * a)
    if not ok:
        return None, status

    if s >= MIN_SCORE and sp < MIN_SCORE and h4_up:
        lo, hi = px - 0.4 * a, px
        sl, tp1, tp2 = lo - 0.8 * a, hi + 1.0 * a, hi + 2.0 * a
        side = "BUY"
    elif s <= -MIN_SCORE and sp > -MIN_SCORE and not h4_up:
        lo, hi = px, px + 0.4 * a
        sl, tp1, tp2 = hi + 0.8 * a, lo - 1.0 * a, lo - 2.0 * a
        side = "SELL"
    else:
        return None, status
    msg = (f"{'🟢' if side == 'BUY' else '🔴'} {side} XAUUSD (A+)\n\n"
           f"Entry: {lo:.0f}-{hi:.0f}\nSL: {sl:.0f}\nTP1: {tp1:.0f}\nTP2: {tp2:.0f}\n\n"
           f"Valid for about 3 hours. Skip it if price already passed the entry zone. "
           f"Risk 1% or less.")
    return msg, status


def main():
    msg, status = analyse()
    print(status)
    if os.getenv("TEST") == "true":
        send("Bot is working.\n" + status)
    if msg:
        send(msg)


if __name__ == "__main__":
    main()
