import datetime as dt
import time
from config import EngineConfig
from run_daily import run_pipeline, send_telegram_alerts


def is_nse_market_open(now: dt.datetime) -> bool:
    if now.weekday() >= 5:  # Sat/Sun
        return False
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end


def watch_loop(interval_minutes: int = 15) -> None:
    cfg = EngineConfig()
    alerted_today = set()
    last_date = dt.date.today()

    print(f"ProQuant Opportunity Watcher started (checking every {interval_minutes} min during NSE hours 09:15-15:30 IST)...")
    while True:
        now = dt.datetime.now()
        if now.date() != last_date:
            alerted_today.clear()
            last_date = now.date()

        if is_nse_market_open(now):
            print(f"[{now.strftime('%H:%M:%S')}] Scanning market for opportunities...")
            calls = run_pipeline(notify=False)
            fresh_buys = [
                c for c in calls
                if c.action in ("STRONG BUY", "BUY") and c.symbol not in alerted_today
            ]
            if fresh_buys:
                send_telegram_alerts(fresh_buys, cfg)
                for c in fresh_buys:
                    alerted_today.add(c.symbol)
        else:
            print(f"[{now.strftime('%H:%M:%S')}] Market closed. Waiting...")

        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    watch_loop(interval_minutes=15)
