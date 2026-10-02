"""
XAUUSD (Gold) A+ bot -> Telegram. Runs hourly on GitHub Actions (free).

Strategies combined (a signal needs enough of them to agree):
 1. Trend following: H4 EMA50/EMA200 trend + H1 EMA50/EMA200
 2. Pullback: Fibonacci 38.2-65% retracement or dip to EMA20
 3. Support/Resistance: swing highs/lows, daily pivot points, $50 round numbers
 4. Price action: pin bar or engulfing candle at the level
 5. Momentum: RSI turn + MACD histogram + ADX strength
Stop loss sits behind the candle/level (+0.5 ATR buffer), 1.0-2.2 ATR wide.
TP1 is capped before the next S/R level; trade is skipped if there is no room.
Not guaranteed. Test on a demo account first.
"""
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TICKERS = ["XAUUSD=X", "GC=F"]   # spot gold first, futures as backup
OFFSET = 0.0        # your MT5 price minus bot price (usually 0 to 2). Set once.
MIN_SCORE = 7       # out of 10. 8 = fewer/stronger, 6 = more signals.
SESSION = (7, 20)   # UTC hours (London + New York)
N = 5               # swing pivot width
MAX_HOLD = 48       # backtest: close trade after this many candles
TOKEN, CHAT = os.getenv("TELEGRAM_TOKEN"), os.getenv("CHAT_ID")


def send(text):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  data={"chat_id": CHAT, "text": text}, timeout=20)


def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def spot():
    try:
        r = requests.get("https://api.gold-api.com/price/XAU", timeout=15).json()
        return float(r["price"])
    except Exception:
        return None


def load():
    for t in TICKERS:
        df = yf.download(t, period="1y", interval="60m", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna().rename(columns=str.lower)
        if len(df) > 400:
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC")
            return df, t
    return None, None


def prepare(df):
    o, h, l, c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
    cs = df["close"]
    d = cs.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).values
    macd = ema(cs, 12) - ema(cs, 26)
    hist = (macd - ema(macd, 9)).values
    pc = cs.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    hd, ld = df.high.diff(), df.low.diff()
    pdm = hd.where((hd > -ld) & (hd > 0), 0.0)
    mdm = (-ld).where((-ld > hd) & (-ld > 0), 0.0)
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1/14, adjust=False).mean() / atr
    adx = (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(alpha=1/14, adjust=False).mean()

    # H4 trend, taken from the last COMPLETED H4 candle (no look-ahead)
    h4 = cs.resample("4h").last().dropna()
    e50h, e200h = ema(h4, 50), ema(h4, 200)
    up4 = ((e50h > e200h) & (h4 > e50h)).astype(float).shift(1)
    dn4 = ((e50h < e200h) & (h4 < e50h)).astype(float).shift(1)
    up4 = (up4.reindex(df.index, method="ffill").fillna(0) > 0.5).values
    dn4 = (dn4.reindex(df.index, method="ffill").fillna(0) > 0.5).values

    # classic daily pivot points from the previous day: P, S1, R1, S2, R2
    day = df.index.normalize()
    dd = df.groupby(day).agg(H=("high", "max"), L=("low", "min"), C=("close", "last")).shift(1)
    P = (dd.H + dd.L + dd.C) / 3
    lv = pd.DataFrame({"P": P, "S1": 2 * P - dd.H, "R1": 2 * P - dd.L,
                       "S2": P - (dd.H - dd.L), "R2": P + (dd.H - dd.L)})
    dp = lv.reindex(day).values

    n = len(df)
    ph = np.array([j for j in range(N, n - N) if h[j] == h[j - N:j + N + 1].max()])
    pl = np.array([j for j in range(N, n - N) if l[j] == l[j - N:j + N + 1].min()])
    W = 12   # major swings (used for targets)
    ph2 = np.array([j for j in range(W, n - W) if h[j] == h[j - W:j + W + 1].max()])
    pl2 = np.array([j for j in range(W, n - W) if l[j] == l[j - W:j + W + 1].min()])
    return dict(ph2=ph2, pl2=pl2, o=o, h=h, l=l, c=c, rsi=rsi, hist=hist, atr=atr.values, adx=adx.values,
                e20=ema(cs, 20).values, e50=ema(cs, 50).values, e200=ema(cs, 200).values,
                up4=up4, dn4=dn4, dp=dp, ph=ph, pl=pl,
                hr=df.index.hour.values, wd=df.index.weekday.values)


def piv(arr, i, w=N):
    """confirmed swing points in the last 400 candles (no look-ahead)"""
    if len(arr) == 0:
        return arr
    return arr[np.searchsorted(arr, max(i - 400, 0)):np.searchsorted(arr, i - w, side="right")]


def candle(D, i, d):
    o, h, l, c = D["o"][i], D["h"][i], D["l"][i], D["c"][i]
    po, pc = D["o"][i - 1], D["c"][i - 1]
    rng, body = h - l, abs(c - o)
    if rng <= 0:
        return None
    if d == 1:
        lw = min(o, c) - l
        if lw >= 2 * body and lw >= 0.5 * rng and c >= l + 0.6 * rng:
            return "bullish pin bar"
        if pc < po and c > po and o <= pc and c > o:
            return "bullish engulfing"
    else:
        uw = h - max(o, c)
        if uw >= 2 * body and uw >= 0.5 * rng and c <= h - 0.6 * rng:
            return "bearish pin bar"
        if pc > po and c < po and o >= pc and c < o:
            return "bearish engulfing"
    return None


def evaluate(D, i):
    """Score both directions on candle i. Returns list of dicts (only the H4-aligned side)."""
    a = D["atr"][i]
    if i < 250 or np.isnan(a) or np.isnan(D["adx"][i]) or np.isnan(D["rsi"][i]):
        return []
    h, l, c = D["h"], D["l"], D["c"]
    sess = SESSION[0] <= D["hr"][i] < SESSION[1] and not (D["wd"][i] == 4 and D["hr"][i] >= 18)
    out = []
    for d in (1, -1):
        if (d == 1 and not D["up4"][i]) or (d == -1 and not D["dn4"][i]):
            continue
        parts, miss, pts = ["H4 trend"], [], 0
        if (D["e50"][i] - D["e200"][i]) * d > 0:
            pts += 1; parts.append("H1 trend")
        # pullback: Fibonacci zone and/or EMA20 dip
        pb = 0
        PH, PL = piv(D["ph"], i), piv(D["pl"], i)
        if len(PH) and len(PL):
            if d == 1:
                jH = PH[-1]; bef = PL[PL < jH]
                if len(bef):
                    leg = h[jH] - l[bef[-1]]
                    if leg > 1.5 * a and 0.382 <= (h[jH] - l[i]) / leg <= 0.65:
                        pb += 1; parts.append("Fib pullback")
            else:
                jL = PL[-1]; bef = PH[PH < jL]
                if len(bef):
                    leg = h[bef[-1]] - l[jL]
                    if leg > 1.5 * a and 0.382 <= (h[i] - l[jL]) / leg <= 0.65:
                        pb += 1; parts.append("Fib pullback")
        if (d == 1 and l[i] <= D["e20"][i] + 0.25 * a and c[i] > D["e50"][i]) or \
           (d == -1 and h[i] >= D["e20"][i] - 0.25 * a and c[i] < D["e50"][i]):
            pb += 1; parts.append("EMA dip")
        pts += pb
        # support / resistance
        tol, lp = 0.4 * a, 0
        probe = l[i] if d == 1 else h[i]
        idx = PL if d == 1 else PH
        hits = int(np.sum(np.abs((l if d == 1 else h)[idx] - probe) <= tol)) if len(idx) else 0
        if hits >= 1: lp += 1; parts.append("swing " + ("support" if d == 1 else "resistance"))
        if hits >= 2: lp += 1; parts.append("tested %dx" % hits)
        if np.any(np.abs(D["dp"][i, [0, 1, 3] if d == 1 else [0, 2, 4]] - probe) <= tol):
            lp += 1; parts.append("daily pivot")
        if abs(probe - round(probe / 50) * 50) <= tol:
            lp += 1; parts.append("round number")
        pts += lp
        cd = candle(D, i, d)
        if cd: pts += 1; parts.append(cd)
        if 35 < D["rsi"][i] < 65 and (D["rsi"][i] - D["rsi"][i - 1]) * d > 0 and \
           (D["hist"][i] - D["hist"][i - 1]) * d > 0:
            pts += 1; parts.append("RSI+MACD turn")
        if D["adx"][i] > 25:
            pts += 1; parts.append("strong ADX")
        if not sess: miss.append("session")
        if D["adx"][i] <= 20: miss.append("trend strength (ADX)")
        if not cd: miss.append("rejection candle")
        if pb < 1: miss.append("pullback")
        if lp < 2: miss.append("support/resistance")
        out.append(dict(d=d, total=pts, ok=not miss and pts >= MIN_SCORE, miss=miss, parts=parts))
    return out


def setup_at(D, i):
    for r in evaluate(D, i):
        if r["ok"]:
            return r
    return None


def build_trade(D, i, d, px):
    a, h, l = D["atr"][i], D["h"], D["l"]
    PH, PL = piv(D["ph2"], i, 12), piv(D["pl2"], i, 12)   # major swing levels only
    if d == 1:
        hi, lo = px, px - 0.3 * a
        sl = min(l[i], l[i - 1]) - 0.5 * a
        risk = hi - sl
        if risk < a: sl, risk = hi - a, a
        cand = list(h[PH][h[PH] > hi + 0.1 * a]) if len(PH) else []
        cand += [v for v in D["dp"][i, [0, 2, 4]] if v > hi + 0.1 * a]
        room = (min(cand) - hi) if cand else 99 * a
    else:
        lo, hi = px, px + 0.3 * a
        sl = max(h[i], h[i - 1]) + 0.5 * a
        risk = sl - lo
        if risk < a: sl, risk = lo + a, a
        cand = list(l[PL][l[PL] < lo - 0.1 * a]) if len(PL) else []
        cand += [v for v in D["dp"][i, [0, 1, 3]] if v < lo - 0.1 * a]
        room = (lo - max(cand)) if cand else 99 * a
    if risk > 2.2 * a or room < 1.1 * risk + 0.1 * a:
        return None   # stop too wide, or next S/R level is too close
    r1 = min(1.5 * risk, room - 0.1 * a)
    if d == 1:
        tp1, tp2 = hi + r1, hi + 2.5 * risk
    else:
        tp1, tp2 = lo - r1, lo - 2.5 * risk
    return dict(lo=lo, hi=hi, sl=sl, tp1=tp1, tp2=tp2, risk=risk)


def backtest(D, idx):
    res, last, n = [], -99, len(D["c"])
    for i in range(250, n - 2):
        if i - last <= 3:
            continue
        r = setup_at(D, i)
        if not r:
            continue
        d = r["d"]
        t = build_trade(D, i, d, D["c"][i])
        if not t:
            continue
        last = i
        entry = t["hi"] if d == 1 else t["lo"]          # worst edge of the zone
        R1 = abs(t["tp1"] - entry) / t["risk"]
        R = None
        for j in range(i + 1, min(i + 1 + MAX_HOLD, n)):
            sl_hit = D["l"][j] <= t["sl"] if d == 1 else D["h"][j] >= t["sl"]
            tp_hit = D["h"][j] >= t["tp1"] if d == 1 else D["l"][j] <= t["tp1"]
            if sl_hit: R = -1.0; break                  # same candle: assume loss
            if tp_hit: R = R1; break
        if R is None:
            R = d * (D["c"][min(i + MAX_HOLD, n - 1)] - entry) / t["risk"]
        res.append(R)
    if not res:
        return "Backtest: no A+ setups in the data."
    r = np.array(res)
    days = (idx[-1] - idx[0]).days
    return (f"Backtest last {days} days (TP1 exit, spread not included):\n"
            f"{len(r)} signals | {np.mean(r > 0) * 100:.0f}% reached TP1 | "
            f"avg {r.mean():+.2f}R | total {r.sum():+.1f}R")


def analyse(test=False):
    full, src = load()
    if full is None:
        return None, "No gold data from Yahoo right now.", ""
    live = spot()
    cur = full["close"].iloc[-1]
    shift = (live - cur) if live and abs(live - cur) < 80 else 0.0
    df = full.iloc[:-1].copy()
    for k in ("open", "high", "low", "close"):
        df[k] = df[k] + shift + OFFSET        # align all levels to spot / your MT5 price
    D = prepare(df)
    i = len(df) - 1
    px = (live if live else cur) + OFFSET
    ev = evaluate(D, i)
    if ev:
        e = ev[0]
        watch = (f"{'BUY' if e['d'] == 1 else 'SELL'} side {e['total']}/10"
                 + (f", waiting for: {', '.join(e['miss'])}" if e["miss"] else ""))
    else:
        watch = "H4 trend unclear, no side to trade"
    status = (f"Gold {px:.1f} | ADX {D['adx'][i]:.0f} | {watch}\n"
              f"src {src}{' +spot' if live else ''}, adj {shift + OFFSET:+.1f}")
    bt = backtest(D, df.index) if test else ""

    r = setup_at(D, i)
    if r and any(((x := setup_at(D, i - k)) and x["d"] == r["d"]) for k in (1, 2, 3)):
        return None, status, bt          # same setup was already sent in the last 3 hours
    if not r:
        return None, status, bt
    t = build_trade(D, i, r["d"], px)
    if not t:
        return None, status + "\nSetup found but stop/targets not good enough, skipped.", bt
    side = "BUY" if r["d"] == 1 else "SELL"
    msg = (f"{'🟢' if r['d'] == 1 else '🔴'} {side} XAUUSD (A+ {r['total']}/10)\n\n"
           f"Entry: {t['lo']:.0f}-{t['hi']:.0f}\nSL: {t['sl']:.0f}\n"
           f"TP1: {t['tp1']:.0f}\nTP2: {t['tp2']:.0f}\n\n"
           f"Price now: {px:.1f}\nWhy: {', '.join(r['parts'])}\n"
           f"Valid about 3 hours. Skip it if price already passed the zone. Risk 1% or less.")
    return msg, status, bt


def main():
    test = os.getenv("TEST") == "true"
    msg, status, bt = analyse(test)
    print(status); print(bt)
    if test:
        send("Bot is working.\n" + status + ("\n\n" + bt if bt else ""))
    if msg:
        send(msg)


if __name__ == "__main__":
    main()
