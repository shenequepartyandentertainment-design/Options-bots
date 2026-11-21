"""
A+ Trade Checklist -> Discord alert bot
Symbols: SPX (via SPY), SPY, SOXS, TSLA

Setup:
    pip install yfinance pandas requests
    export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/XXXX/YYYY"
    python aplus_alert_bot.py

Notes:
- yfinance data is free but ~15 min delayed for some tickers. For real-time,
  swap get_bars() for Alpaca / Polygon / Tradier (same DataFrame shape).
- SPX has no volume, so SPX is evaluated on SPY (they move together).
- Checklist items 8 (option spread) and 9 (stop/target/max loss) can't be
  verified from stock data, so they're included as reminders in the alert.
- Alerts only. It never places trades. Not financial advice.
"""

import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
SYMBOLS = {"SPX": "SPY", "SPY": "SPY", "SOXS": "SOXS", "TSLA": "TSLA"}  # label -> data ticker
SCAN_EVERY_SEC = 60
COOLDOWN_MIN = 30          # don't repeat the same alert for this long
MIN_SCORE = 7              # of 7 auto-checks; lower to 6 for more alerts
VOLUME_MULT = 1.0          # confirmation bar volume vs 20-bar average
LEVEL_TOLERANCE_ATR = 0.35 # how close to a level counts as "pulled back to it"
NY = ZoneInfo("America/New_York")

last_alert = {}


def market_open(now):
    if now.weekday() >= 5:
        return False
    t = now.time()
    return datetime(2000, 1, 1, 9, 35).time() <= t <= datetime(2000, 1, 1, 15, 45).time()


def get_bars(ticker, interval, period="5d"):
    df = yf.Ticker(ticker).history(period=period, interval=interval, prepost=False)
    if df.empty:
        return df
    df.index = df.index.tz_convert(NY)
    # drop the still-forming candle so we only use CLOSED bars
    mins = int(interval.replace("m", ""))
    now = datetime.now(NY)
    if df.index[-1] + timedelta(minutes=mins) > now:
        df = df.iloc[:-1]
    return df


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def session_vwap(df):
    today = df[df.index.date == df.index[-1].date()]
    tp = (today["High"] + today["Low"] + today["Close"]) / 3
    return (tp * today["Volume"]).cumsum() / today["Volume"].cumsum()


def evaluate(ticker):
    d5 = get_bars(ticker, "5m")
    d15 = get_bars(ticker, "15m")
    if len(d5) < 40 or len(d15) < 30:
        return None

    # ---- 15m trend ----
    e9_15, e21_15 = ema(d15["Close"], 9), ema(d15["Close"], 21)
    up15 = d15["Close"].iloc[-1] > e21_15.iloc[-1] and e9_15.iloc[-1] > e21_15.iloc[-1]
    dn15 = d15["Close"].iloc[-1] < e21_15.iloc[-1] and e9_15.iloc[-1] < e21_15.iloc[-1]

    # ---- 5m data ----
    c = d5["Close"]
    e9, e21 = ema(c, 9), ema(c, 21)
    vwap = session_vwap(d5)
    a = atr(d5).iloc[-1]
    last = d5.iloc[-1]                      # last CLOSED 5m candle
    today = d5[d5.index.date == d5.index[-1].date()]
    prev = d5[d5.index.date < d5.index[-1].date()]

    # meaningful levels: VWAP, 21 EMA, prior day high/low, opening range high/low
    levels = [vwap.iloc[-1], e21.iloc[-1]]
    if len(prev):
        pd_day = prev[prev.index.date == prev.index[-1].date()]
        levels += [pd_day["High"].max(), pd_day["Low"].min()]
    orb = today.iloc[:6]  # first 30 min
    if len(orb):
        levels += [orb["High"].max(), orb["Low"].min()]
    tol = LEVEL_TOLERANCE_ATR * a
    recent = d5.iloc[-6:]

    avg_vol = d5["Volume"].iloc[-21:-1].mean()
    results = {}
    for side in ("LONG", "SHORT"):
        L = side == "LONG"
        checks = {
            "15m trend agrees": up15 if L else dn15,
            "5m confirms direction": (c.iloc[-1] > c.iloc[-4]) if L else (c.iloc[-1] < c.iloc[-4]),
            "Correct side of VWAP": (last["Close"] > vwap.iloc[-1]) if L else (last["Close"] < vwap.iloc[-1]),
            "9 EMA / 21 EMA agree": (e9.iloc[-1] > e21.iloc[-1]) if L else (e9.iloc[-1] < e21.iloc[-1]),
            "Pulled back to level": any(
                (recent["Low"] <= lv + tol).any() and (recent["Low"] >= lv - tol).any() if L
                else (recent["High"] >= lv - tol).any() and (recent["High"] <= lv + tol).any()
                for lv in levels
            ),
            "Confirmation candle closed": (
                (last["Close"] > last["Open"] and last["Close"] > e9.iloc[-1]) if L
                else (last["Close"] < last["Open"] and last["Close"] < e9.iloc[-1])
            ),
            "Volume supports move": last["Volume"] >= VOLUME_MULT * avg_vol,
        }
        results[side] = checks

    return {"price": float(last["Close"]), "atr": float(a), "low": float(last["Low"]),
            "high": float(last["High"]), "results": results, "bar_time": d5.index[-1]}


def send_discord(label, side, data, checks):
    price, a = data["price"], data["atr"]
    if side == "LONG":
        stop = min(data["low"], price - 1.0 * a)
        target = price + 2 * (price - stop)
        idea = "CALLS"
    else:
        stop = max(data["high"], price + 1.0 * a)
        target = price - 2 * (stop - price)
        idea = "PUTS"
    lines = "\n".join(f"✅ {k}" for k, v in checks.items() if v)
    missed = "\n".join(f"❌ {k}" for k, v in checks.items() if not v)
    embed = {
        "title": f"🚨 {label} A+ setup: {side} ({idea})",
        "color": 0x2ECC71 if side == "LONG" else 0xE74C3C,
        "description": (
            f"**Score: {sum(checks.values())}/7** | Bar closed {data['bar_time'].strftime('%H:%M')} ET\n\n"
            f"{lines}\n{missed}\n\n"
            f"**Underlying ref ({SYMBOLS[label]})**\n"
            f"Entry ≈ {price:.2f} | Stop ≈ {stop:.2f} | Target (2R) ≈ {target:.2f}\n\n"
            f"**Still on YOU:**\n"
            f"⬜ Check the option bid/ask spread\n"
            f"⬜ Confirm your stop, target and max possible loss"
        ),
    }
    r = requests.post(WEBHOOK_URL, json={"embeds": [embed]}, timeout=10)
    r.raise_for_status()


def scan():
    now = datetime.now(NY)
    if not market_open(now):
        return
    for label, ticker in SYMBOLS.items():
        try:
            data = evaluate(ticker)
            if not data:
                continue
            # stateless runs: only alert if the signal bar closed in the last ~8 min
            closed_at = data["bar_time"] + timedelta(minutes=5)
            if now - closed_at > timedelta(minutes=8):
                continue
            for side, checks in data["results"].items():
                score = sum(checks.values())
                key = (label, side)
                if score >= MIN_SCORE and now - last_alert.get(key, datetime.min.replace(tzinfo=NY)) > timedelta(minutes=COOLDOWN_MIN):
                    send_discord(label, side, data, checks)
                    last_alert[key] = now
                    print(f"[{now:%H:%M}] alert sent: {label} {side} {score}/7")
        except Exception as e:
            print(f"[{now:%H:%M}] error on {label}: {e}")


if __name__ == "__main__":
    if not WEBHOOK_URL:
        raise SystemExit("Set DISCORD_WEBHOOK_URL first.")
    scan()  # single scan per run (GitHub Actions schedules the repeats)
