"""
XAUUSD (Gold) A+ bot -> Telegram, with live trade management + weekly report.
Runs every 15 min on GitHub Actions (free). State is kept in trades.json.

Signal = trend (H4/H1) + pullback (Fib/EMA20) + support/resistance (swings, daily
pivots, $50 round numbers) + candle (pin bar/engulfing) + momentum (RSI/MACD/ADX).
Plan: close 1/3 at TP1 (move SL to entry), 1/3 at TP2, last 1/3 at TP3.
1 pip = $0.10 on gold (50 pips = $5). Not guaranteed. Demo account first.
"""
import os
import json
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TICKERS = ["XAUUSD=X", "GC=F"]   # spot gold first, futures as backup
OFFSET = 0.0        # your MT5 price minus bot price (usually 0 to 2). Set once.
MIN_SCORE = 6       # out of 10. 7-8 = fewer/stronger, 5 = more signals.
SESSION = (7, 20)   # UTC hours (London + New York)
N = 5               # swing pivot width
MAX_HOLD = 48       # backtest: close trade after this many candles
STATE_FILE = "trades.json"
# Big news (UTC). No new signals +-90 min, open trades get a warning. Add dates yourself
# from forexfactory.com (NFP, FOMC, CPI). The first one is the usual US jobs report: CHECK it.
NEWS_UTC = ["2026-10-02 12:30"]
TOKEN, CHAT = os.getenv("TELEGRAM_TOKEN"), os.getenv("CHAT_ID")


def send(text):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  data={"chat_id": CHAT, "text": text}, timeout=20)


def now_utc(): return pd.Timestamp.now(tz="UTC")
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def pips(a, b): return abs(a - b) * 10


def spot():
    try:
        return float(requests.get("https://api.gold-api.com/price/XAU", timeout=15).json()["price"])
    except Exception:
        return None


def _clean(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().rename(columns=str.lower)
    if len(df) and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC")
    return df


def load():
    for t in TICKERS:
        df = _clean(yf.download(t, period="1y", interval="60m", progress=False, auto_adjust=True))
        if len(df) > 400:
            return df, t
    return None, None


def load15(t, adj):
    try:
        m = _clean(yf.download(t, period="5d", interval="15m", progress=False, auto_adjust=True)).iloc[:-1]
        for k in ("open", "high", "low", "close"):
            m[k] = m[k] + adj
        return m
    except Exception:
        return None


# ---------------------------------------------------------------- indicators
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

    h4 = cs.resample("4h").last().dropna()          # last COMPLETED H4 candle (no look-ahead)
    e50h, e200h = ema(h4, 50), ema(h4, 200)
    up4 = ((e50h > e200h) & (h4 > e50h)).astype(float).shift(1)
    dn4 = ((e50h < e200h) & (h4 < e50h)).astype(float).shift(1)
    up4 = (up4.reindex(df.index, method="ffill").fillna(0) > 0.5).values
    dn4 = (dn4.reindex(df.index, method="ffill").fillna(0) > 0.5).values

    day = df.index.normalize()                      # daily pivots from previous day
    dd = df.groupby(day).agg(H=("high", "max"), L=("low", "min"), C=("close", "last")).shift(1)
    P = (dd.H + dd.L + dd.C) / 3
    dp = pd.DataFrame({"P": P, "S1": 2 * P - dd.H, "R1": 2 * P - dd.L,
                       "S2": P - (dd.H - dd.L), "R2": P + (dd.H - dd.L)}).reindex(day).values

    n = len(df)
    def sw(w, hi):
        a_ = h if hi else l
        f = np.max if hi else np.min
        return np.array([j for j in range(w, n - w) if a_[j] == f(a_[j - w:j + w + 1])])
    return dict(o=o, h=h, l=l, c=c, rsi=rsi, hist=hist, atr=atr.values, adx=adx.values,
                e20=ema(cs, 20).values, e50=ema(cs, 50).values, e200=ema(cs, 200).values,
                up4=up4, dn4=dn4, dp=dp, ph=sw(N, True), pl=sw(N, False),
                ph2=sw(12, True), pl2=sw(12, False),
                hr=df.index.hour.values, wd=df.index.weekday.values)


def piv(arr, i, w=N):
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
        if lw >= 2 * body and lw >= 0.5 * rng and c >= l + 0.6 * rng: return "bullish pin bar"
        if pc < po and c > po and o <= pc and c > o: return "bullish engulfing"
    else:
        uw = h - max(o, c)
        if uw >= 2 * body and uw >= 0.5 * rng and c <= h - 0.6 * rng: return "bearish pin bar"
        if pc > po and c < po and o >= pc and c < o: return "bearish engulfing"
    return None


def evaluate(D, i):
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


def pick(ev):
    for r in ev:
        if r["ok"]:
            return r
    return None


def ahead(D, i, d, ref):
    """nearest major support/resistance level beyond ref in trade direction"""
    PH, PL = piv(D["ph2"], i, 12), piv(D["pl2"], i, 12)
    if d == 1:
        c_ = list(D["h"][PH][D["h"][PH] > ref]) if len(PH) else []
        c_ += [v for v in D["dp"][i, [0, 2, 4]] if v > ref]
        return min(c_) if c_ else None
    c_ = list(D["l"][PL][D["l"][PL] < ref]) if len(PL) else []
    c_ += [v for v in D["dp"][i, [0, 1, 3]] if v < ref]
    return max(c_) if c_ else None


def build_trade(D, i, d, px):
    a, h, l = D["atr"][i], D["h"], D["l"]
    if d == 1:
        hi, lo = px, px - 0.3 * a
        sl = min(l[i], l[i - 1]) - 0.5 * a
        risk = hi - sl
        if risk < a: sl, risk = hi - a, a
        lv = ahead(D, i, 1, hi + 0.1 * a); room = (lv - hi) if lv else 99 * a
    else:
        lo, hi = px, px + 0.3 * a
        sl = max(h[i], h[i - 1]) + 0.5 * a
        risk = sl - lo
        if risk < a: sl, risk = lo + a, a
        lv = ahead(D, i, -1, lo - 0.1 * a); room = (lo - lv) if lv else 99 * a
    if risk > 2.2 * a or room < 1.1 * risk + 0.1 * a:
        return None   # stop too wide, or next S/R level too close
    e = hi if d == 1 else lo
    r1 = min(1.5 * risk, room - 0.1 * a)
    return dict(lo=lo, hi=hi, sl=sl, risk=risk, entry=e,
                tp1=e + d * r1, tp2=e + d * 2.5 * risk, tp3=e + d * 3.5 * risk)


# ---------------------------------------------------------------- estimate
def excursions(D, EV):
    """how far (in ATR) similar past setups moved in our favour before a 1.5 ATR adverse move"""
    mf, last, n = [], -99, len(D["c"])
    for j in range(250, n - 26):
        if j - last <= 3:
            continue
        r = [x for x in EV[j] if x["total"] >= MIN_SCORE - 2]
        if not r:
            continue
        last, dj, e, aj, m = j, r[0]["d"], D["c"][j], D["atr"][j], 0.0
        for k in range(j + 1, j + 25):
            if ((e - D["l"][k]) if dj == 1 else (D["h"][k] - e)) > 1.5 * aj:
                break
            m = max(m, (D["h"][k] - e) if dj == 1 else (e - D["l"][k]))
        mf.append(m / aj)
    return np.array(mf)


def estimate(mf, a, ref, t):
    r5 = lambda x: int(round(x / 5.0) * 5)
    if len(mf) >= 15:
        lo, hi = r5(np.percentile(mf, 40) * a * 10), r5(np.percentile(mf, 60) * a * 10)
        pr = [float(np.mean(mf >= abs(t["tp%d" % k] - ref) / a)) for k in (1, 2, 3)]
        basis = f"{len(mf)} milte julte setups"
    else:
        lo, hi, pr, basis = r5(a * 10), r5(a * 18), None, "ATR ka andaza, data kam hai"
    cap = r5(pips(t["tp3"], ref) * 1.1)
    lo, hi = min(lo, cap), min(hi, cap)
    return [lo, max(hi, lo + 5)], pr, basis


# ---------------------------------------------------------------- trade lifecycle
def result_R(tr, close=None):
    R = [abs(tr["tp%d" % k] - tr["entry"]) / tr["risk"] for k in (1, 2, 3)]
    if tr.get("done") == "SL":
        return -1.0
    h = tr["hit"]
    base = sum(R[:h]) / 3
    if close is not None:
        base += (3 - h) / 3 * tr["d"] * (close - tr["entry"]) / tr["risk"]
    return base


def step(tr, ts, H, L, out):
    """feed one price range (high, low) into a trade. Returns True when the trade is finished."""
    d, ref = tr["d"], tr["ref"]
    if not tr["filled"]:
        if ts > pd.Timestamp(tr["t0"]) + pd.Timedelta(hours=3):
            tr["done"] = "expired"
            out.append("⌛ Signal expire: price entry zone mein nahi aya. Ye trade skip karein.")
            return True
        if L <= tr["hi"] and H >= tr["lo"]:
            tr["filled"], tr["fill_t"] = True, ts.isoformat()
            out.append(f"🔔 Entry zone touch hua ({tr['lo']:.0f}-{tr['hi']:.0f}). Trade active hai, SL {tr['sl']:.0f}.")
        else:
            return False
    stop = tr["entry"] if tr["be"] else tr["sl"]
    if (L <= stop) if d == 1 else (H >= stop):
        if tr["be"]:
            tr["done"] = "BE"
            out.append(f"➖ Price entry par wapas aa gaya: trade breakeven par band (TP{tr['hit']} ka profit mehfooz).")
        else:
            tr["done"] = "SL"
            out.append(f"❌ SL hit: -{pips(tr['sl'], ref):.0f} pips. Trade band. Agle signal ka intezar karein.")
        return True
    for k in (1, 2, 3):
        if tr["hit"] >= k:
            continue
        tp = tr["tp%d" % k]
        if not ((H >= tp) if d == 1 else (L <= tp)):
            break
        tr["hit"], tr["be"] = k, True
        gain = f"+{pips(tp, ref):.0f} pips"
        if k == 1:
            out.append(f"✅ TP1 hit ({gain}).\nAb: 1/3 position band karein aur SL ko entry ({tr['entry']:.0f}) par le aayen.\n"
                       f"Estimate: ~{tr['est'][0]}-{tr['est'][1]} pips tak ja sakta hai, wahan tak chalne dein.")
        elif k == 2:
            out.append(f"✅✅ TP2 hit ({gain}).\nAb: aur 1/3 band karein, baqi ko TP3 ({tr['tp3']:.0f}) tak chalne dein (SL entry par).")
        else:
            tr["done"] = "TP3"
            out.append(f"🏆 TP3 hit ({gain})! Poora trade band karein. Shandar.")
            return True
    fav = (H if d == 1 else L)
    tr["best"] = max(tr["best"], fav) if d == 1 else min(tr["best"], fav)
    return False


def flag(tr, name, msg, out):
    if name not in tr["flags"]:
        tr["flags"].append(name)
        out.append(msg)


def advise(tr, p, now, D, i, a, out):
    d, ent = tr["d"], tr["entry"]
    if (d == 1 and not D["up4"][i]) or (d == -1 and not D["dn4"][i]):
        flag(tr, "trend", "⚠️ H4 trend kamzor/ulta ho raha hai. Risk barh gaya: lot kam karein ya profit mein ho to band karne ka soch lein.", out)
    lv = ahead(D, i, d, p)
    if lv is not None and abs(lv - p) < 0.3 * a and (p - ent) * d > 0.8 * a:
        flag(tr, "level", f"⚠️ Price bade level {lv:.0f} ke qareeb hai. Yahan se palat sakta hai: profit lena behtar, ya zyada tar position band karein.", out)
    if tr["hit"] >= 1:
        best = (tr["best"] - ent) * d
        if best > 0 and (p - ent) * d < 0.5 * best:
            flag(tr, "give", "⚠️ Profit wapas ja raha hai (aadha ghat gaya). Band karne par ghaur karein.", out)
    if tr["hit"] == 0 and tr.get("fill_t") and now - pd.Timestamp(tr["fill_t"]) > pd.Timedelta(hours=12):
        flag(tr, "old", "⏳ Trade 12 ghante se zyada purana hai aur TP1 nahi aya. Breakeven ya chote nuksan par nikalne ka soch lein.", out)
    if now.weekday() == 4 and now.hour >= 18:
        flag(tr, "wkend", "🛑 Weekend qareeb hai (gap ka risk). Trade band kar dein.", out)
    elif now.hour >= 19:
        flag(tr, "sess", "🌙 Session khatam hone wala hai, volume kam hoga. SL entry par rakhein ya trade band karein.", out)
    for nw in NEWS_UTC:
        dt = (pd.Timestamp(nw, tz="UTC") - now).total_seconds() / 60
        if 0 <= dt <= 60:
            flag(tr, "news" + nw, f"📰 Bari news {nw} UTC par hai. Lot kam karein ya trade band karein, gold bohat uchhalta hai.", out)


def news_near(now):
    return any(abs((pd.Timestamp(n, tz="UTC") - now).total_seconds()) <= 5400 for n in NEWS_UTC)


# ---------------------------------------------------------------- backtest
def backtest(D, EV, idx):
    res, last, n = [], -99, len(D["c"])
    for i in range(250, n - 2):
        if i - last <= 3:
            continue
        r = pick(EV[i])
        t = build_trade(D, i, r["d"], D["c"][i]) if r else None
        if not t:
            continue
        last = i
        tr = dict(t, d=r["d"], filled=True, be=False, hit=0, best=t["entry"], ref=t["entry"],
                  est=[0, 0], flags=[], t0=idx[i].isoformat())
        fin, end = False, min(i + 1 + MAX_HOLD, n)
        for j in range(i + 1, end):
            if step(tr, idx[j], D["h"][j], D["l"][j], []):
                fin = True; break
        res.append(result_R(tr, None if fin else D["c"][end - 1]))
    if not res:
        return "Backtest: koi A+ setup nahi mila."
    r = np.array(res)
    return (f"Backtest pichle {(idx[-1] - idx[0]).days} din (1/3-1/3-1/3 plan, spread shamil nahi):\n"
            f"{len(r)} signals | {np.mean(r > 0) * 100:.0f}% profit mein | avg {r.mean():+.2f}R | total {r.sum():+.1f}R")


# ---------------------------------------------------------------- weekly report
def wk_of(iso):
    c = pd.Timestamp(iso).isocalendar()
    return f"{c[0]}-W{c[1]:02d}"


def report(state, wk):
    ts = [t for t in state["closed"] + state["open"] if wk_of(t["t0"]) == wk]
    if not ts:
        return f"📊 Weekly report {wk}\nIs hafte koi signal nahi aya."
    dn = lambda t: t.get("done")
    win = [t for t in ts if dn(t) == "TP3" or (dn(t) == "BE" and t["hit"] >= 2)]
    be1 = [t for t in ts if dn(t) == "BE" and t["hit"] == 1]
    loss = [t for t in ts if dn(t) == "SL"]
    exp = [t for t in ts if dn(t) == "expired"]
    opn = [t for t in ts if dn(t) is None]
    fin = win + be1 + loss
    acc = f"{(len(win) + len(be1)) / len(fin) * 100:.0f}% ({len(win) + len(be1)}/{len(fin)})" if fin else "abhi koi band trade nahi"
    R = sum(result_R(t) for t in fin)
    pp = sum(result_R(t) * t["risk"] * 10 for t in fin)
    lab = {"TP3": "TP3 poora", "BE": "TP%d phir breakeven", "SL": "SL", "expired": "entry nahi mili", None: "abhi open"}
    lines = []
    for t in ts:
        nm = lab[dn(t)] % t["hit"] if dn(t) == "BE" else lab[dn(t)]
        lines.append(f"{pd.Timestamp(t['t0']).strftime('%a %d %b')} {'BUY' if t['d'] == 1 else 'SELL'}: {nm}")
    return (f"📊 Weekly report {wk}\n"
            f"Signals: {len(ts)}\n✅ Wins (TP2/TP3 tak): {len(win)}\n➖ TP1 ke baad breakeven: {len(be1)}\n"
            f"❌ Losses (SL): {len(loss)}\n⌛ Entry nahi mili: {len(exp)}\n🔄 Abhi open: {len(opn)}\n\n"
            f"Accuracy (TP1 tak pahunche): {acc}\n"
            f"Net: {R:+.2f}R, lagbhag {pp:+.0f} pips (spread shamil nahi)\n\n" + "\n".join(lines))


# ---------------------------------------------------------------- main run
def load_state():
    try:
        s = json.load(open(STATE_FILE))
        s.setdefault("open", []); s.setdefault("closed", []); s.setdefault("last_candle", ""); s.setdefault("last_report", "")
        return s
    except Exception:
        return {"open": [], "closed": [], "last_candle": "", "last_report": ""}


def run(test=False):
    state, out = load_state(), []
    full, src = load()
    if full is None:
        return ["Gold ka data abhi Yahoo se nahi mila."], state
    live = spot()
    cur = full["close"].iloc[-1]
    shift = (live - cur) if live and abs(live - cur) < 80 else 0.0
    adj = shift + OFFSET
    df = full.iloc[:-1].copy()
    for k in ("open", "high", "low", "close"):
        df[k] = df[k] + adj                      # all levels aligned to spot / your MT5 price
    D = prepare(df)
    EV = [evaluate(D, j) for j in range(len(df))]
    i, now = len(df) - 1, now_utc()
    a = D["atr"][i]
    px = (live if live else cur) + OFFSET
    m15 = load15(src, adj)

    # 1) manage open trades
    for tr in list(state["open"]):
        chk = pd.Timestamp(tr["chk"])
        rows = []
        if m15 is not None:
            rows = [(ts, r.high, r.low) for ts, r in m15.iterrows() if ts > chk]
            if rows:
                tr["chk"] = rows[-1][0].isoformat()
        if live:
            rows.append((now, px, px))
        for ts, H, L in rows:
            if step(tr, ts, H, L, out):
                break
        if tr.get("done"):
            state["open"].remove(tr); state["closed"].append(tr)
        elif tr["filled"]:
            advise(tr, px, now, D, i, a, out)

    # 2) new signal (once per closed H1 candle, one trade at a time)
    status_note = ""
    cid = df.index[-1].isoformat()
    if cid != state["last_candle"]:
        state["last_candle"] = cid
        r = pick(EV[i])
        if r and state["open"]:
            status_note = "signal mila lekin purana trade abhi open hai, skip."
        elif r and news_near(now):
            status_note = "signal mila lekin bari news qareeb hai, skip."
        elif r:
            t = build_trade(D, i, r["d"], px)
            if not t:
                status_note = "setup mila lekin SL/targets achhe nahi the, skip."
            else:
                d = r["d"]
                ref = (t["lo"] + t["hi"]) / 2
                est, pr, basis = estimate(excursions(D, EV), a, ref, t)
                tr = dict(id=cid, d=d, t0=(df.index[-1] + pd.Timedelta(hours=1)).isoformat(),
                          lo=float(t["lo"]), hi=float(t["hi"]), sl=float(t["sl"]), risk=float(t["risk"]),
                          entry=float(t["entry"]), tp1=float(t["tp1"]), tp2=float(t["tp2"]), tp3=float(t["tp3"]),
                          ref=float(ref), est=est, filled=False, be=False, hit=0, best=float(t["entry"]),
                          flags=[], chk=now.isoformat())
                state["open"].append(tr)
                chance = (" | ".join(f"TP{k + 1} {pr[k] * 100:.0f}%" for k in range(3)) if pr else "data kam")
                out.append(
                    f"{'🟢' if d == 1 else '🔴'} {'BUY' if d == 1 else 'SELL'} XAUUSD (A+ {r['total']}/10)\n\n"
                    f"Entry: {t['lo']:.0f}-{t['hi']:.0f}\n"
                    f"SL: {t['sl']:.0f} (-{pips(t['sl'], ref):.0f} pips)\n"
                    f"TP1: {t['tp1']:.0f} ({pips(t['tp1'], ref):.0f} pips)\n"
                    f"TP2: {t['tp2']:.0f} ({pips(t['tp2'], ref):.0f} pips)\n"
                    f"TP3: {t['tp3']:.0f} ({pips(t['tp3'], ref):.0f} pips)\n\n"
                    f"Estimate: aam taur par ~{est[0]}-{est[1]} pips tak chalta hai ({basis}).\n"
                    f"Chance (pichle setups ke hisaab se, guarantee nahi): {chance}\n"
                    f"Plan: TP1 par 1/3 band + SL entry par, TP2 par 1/3, baqi TP3 ya estimate par.\n\n"
                    f"Price now: {px:.1f}\nWhy: {', '.join(r['parts'])}\n"
                    f"3 ghante mein entry na mile to skip karein. Risk 1% se kam.")

    # 3) weekly report: Friday after 21:00 UTC
    wk = wk_of(now.isoformat())
    if now.weekday() == 4 and now.hour >= 21 and state["last_report"] != wk:
        state["last_report"] = wk
        out.append(report(state, wk))

    if test:
        ev = EV[i]
        watch = ((f"{'BUY' if ev[0]['d'] == 1 else 'SELL'} side {ev[0]['total']}/10"
                  + (f", intezar: {', '.join(ev[0]['miss'])}" if ev[0]["miss"] else "")) if ev
                 else "H4 trend saaf nahi, koi side nahi")
        out.insert(0, f"Bot is working.\nGold {px:.1f} | ADX {D['adx'][i]:.0f} | {watch}\n"
                      f"src {src}{' +spot' if live else ''}, adj {adj:+.1f}\n"
                      f"Open trades: {len(state['open'])} {status_note}\n\n{backtest(D, EV, df.index)}")
        out.append(report(state, wk))
    return out, state


def main():
    out, state = run(os.getenv("TEST") == "true")
    for m in out:
        print(m); print("-" * 20)
        send(m)
    json.dump(state, open(STATE_FILE, "w"), indent=1)


if __name__ == "__main__":
    main()
