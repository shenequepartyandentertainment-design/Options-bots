import os
import time
import logging
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

SCAN_INTERVAL_SEC = 300
WORKERS = 8
MAX_EXPIRATIONS = 3
MIN_NEW_VOLUME = 2000
MIN_NEW_PREMIUM = 250_000
DOMINANCE_RATIO = 2.0
UNUSUAL_MIN_VOLUME = 1000
UNUSUAL_VOL_OI_RATIO = 3.0
ALERT_COOLDOWN_MIN = 60
TZ = ZoneInfo("America/New_York")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("flowbot")

state = {}
last_alert = {}


def load_tickers():
    if os.path.exists("tickers.txt"):
        with open("tickers.txt") as f:
            return sorted({l.strip().upper() for l in f if l.strip()})
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return [s.replace(".", "-") for s in df["Symbol"].tolist()]
    except Exception:
        return ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL"]


def market_open(now=None):
    now = now or dt.datetime.now(TZ)
    if now.weekday() >= 5:
        return False
    return dt.time(9, 30) <= now.time() <= dt.time(16, 0)


def send_alert(msg):
    log.info("ALERT: %s", msg.replace("\n", " | "))
    hook = os.getenv("DISCORD_WEBHOOK")
    try:
        if hook:
            requests.post(hook, json={"content": msg}, timeout=10)
    except Exception as e:
        log.warning("alert send failed: %s", e)


def fetch_flow(sym):
    t = yf.Ticker(sym)
    exps = t.options[:MAX_EXPIRATIONS]
    out = {"cv": 0, "pv": 0, "cp": 0.0, "pp": 0.0, "unusual": []}
    for exp in exps:
        ch = t.option_chain(exp)
        for side, df in (("CALL", ch.calls), ("PUT", ch.puts)):
            df = df.fillna(0)
            vol = df["volume"]
            prem = (vol * df["lastPrice"] * 100).sum()
            if side == "CALL":
                out["cv"] += int(vol.sum()); out["cp"] += prem
            else:
                out["pv"] += int(vol.sum()); out["pp"] += prem
            hot = df[(vol >= UNUSUAL_MIN_VOLUME) &
                     (vol >= UNUSUAL_VOL_OI_RATIO * df["openInterest"].clip(lower=1))]
            for _, r in hot.iterrows():
                out["unusual"].append(
                    f"{side} {r['strike']:g} {exp} vol {int(r['volume']):,} "
                    f"OI {int(r['openInterest']):,}")
    return out


def cooled(sym, kind):
    now = dt.datetime.now()
    prev = last_alert.get((sym, kind))
    if prev and (now - prev).total_seconds() < ALERT_COOLDOWN_MIN * 60:
        return False
    last_alert[(sym, kind)] = now
    return True


def evaluate(sym, cur):
    prev = state.get(sym)
    state[sym] = cur
    alerts = []
    if prev:
        dcv, dpv = cur["cv"] - prev["cv"], cur["pv"] - prev["pv"]
        dcp, dpp = cur["cp"] - prev["cp"], cur["pp"] - prev["pp"]
        if dcv >= MIN_NEW_VOLUME and dcp >= MIN_NEW_PREMIUM and dcv >= DOMINANCE_RATIO * max(dpv, 1):
            if cooled(sym, "calls"):
                alerts.append(f"🟢 BIG CALLS {sym}\n+{dcv:,} contracts (~${dcp:,.0f}) vs puts +{dpv:,}")
        if dpv >= MIN_NEW_VOLUME and dpp >= MIN_NEW_PREMIUM and dpv >= DOMINANCE_RATIO * max(dcv, 1):
            if cooled(sym, "puts"):
                alerts.append(f"🔴 BIG PUTS {sym}\n+{dpv:,} contracts (~${dpp:,.0f}) vs calls +{dcv:,}")
        if dcv >= MIN_NEW_VOLUME and dpv >= MIN_NEW_VOLUME and dcp >= MIN_NEW_PREMIUM and dpp >= MIN_NEW_PREMIUM:
            if cooled(sym, "both"):
                alerts.append(f"🟡 BIG CALLS + PUTS {sym}\ncalls +{dcv:,} (~${dcp:,.0f}) | puts +{dpv:,} (~${dpp:,.0f})")
    if cur["unusual"] and cooled(sym, "unusual"):
        alerts.append(f"🔥 UNUSUAL {sym}\n" + "\n".join(cur["unusual"][:5]))
    return alerts


def scan_once(tickers):
    n_alerts = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_flow, s): s for s in tickers}
        for f in as_completed(futs):
            sym = futs[f]
            try:
                for a in evaluate(sym, f.result()):
                    send_alert(a); n_alerts += 1
            except Exception as e:
                log.debug("%s skipped: %s", sym, e)
    return n_alerts


def main():
    tickers = load_tickers()
    log.info("Loaded %d tickers", len(tickers))
    send_alert(f"✅ Options flow bot started, watching {len(tickers)} tickers")
    while True:
        try:
            if market_open():
                start = time.time()
                n = scan_once(tickers)
                log.info("Pass done in %.0fs, %d alerts", time.time() - start, n)
                time.sleep(SCAN_INTERVAL_SEC)
            else:
                if os.getenv("EXIT_AT_CLOSE") and state:
                    log.info("Market closed, exiting")
                    break
                state.clear()
                log.info("Market closed, sleeping")
                time.sleep(60)
        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error("loop error: %s", e)
            time.sleep(60)


if __name__ == "__main__":
    main()
