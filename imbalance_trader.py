import threading
import time
from datetime import datetime, time as dtime, timedelta
import pytz
import csv
import os
import subprocess
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import socket
import pandas as pd
from collections import deque
import numpy as np


# ------------------ Config ------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "data", "imbalance_prices.csv")
TRADE_LOG_FILE = os.path.join(BASE_DIR, "data", "trading_log.csv")
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

FAST_WINDOW = 1577910
SLOW_WINDOW  = 1673681
TRADE_FEE   = 0.0003

PT = pytz.timezone("US/Pacific")
ET = pytz.timezone("US/Eastern")


# For holding the currently active IBApp (for threading compatibility)
app_holder = {'app': None}

CLIENT_ID = 2

def make_ib_app_instance():
    class IBApp(EWrapper, EClient):
        def __init__(self, host="127.0.0.1", port=4002, client_id=1):
            EClient.__init__(self, self)
            self.host = host
            self.port = port
            self.client_id = client_id


            # Latest tick data
            self.bid_size = None
            self.ask_size = None
            self.last_price = None


            # Market hours
            tomorrow = (datetime.now(PT) + timedelta(days=1)).date()
            self.pst_start = PT.localize(datetime.combine(tomorrow, dtime(6, 30, 00)))
            self.pst_end = PT.localize(datetime.combine(tomorrow, dtime(12, 59, 59)))
            self._hours_set = True


            # Threading
            self._lock = threading.RLock()
            self._subscribed = False
        
        def _log_trade(self, timestamp, side, price, pct):
            file_exists = os.path.exists(TRADE_LOG_FILE)

            with open(TRADE_LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(["timestamp", "side", "price", "pct_change"])

                writer.writerow([
                    timestamp,
                    side,
                    f"{price:.4f}",
                    f"{pct:.6f}"
                ])


        # ------------------ Market Data ------------------ #
        def tickSize(self, tickerId, field, size):
            size = float(size)

            if field == 0:  # Bid size
                self.bid_size = size
                self.record_tick()
            elif field == 3:  # Ask size
                self.ask_size = size
                self.record_tick()


        def tickPrice(self, tickerId, field, price, attrib):
            if field == 1:
                self.last_bid_price = price
            elif field == 2:
                self.last_ask_price = price
            elif field == 4:
                self.last_price = price


        def record_tick(self):
            if self.bid_size is None or self.ask_size is None or self.last_price is None:
                return
            if abs(self.bid_size) > 1e12 or abs(self.ask_size) > 1e12:
                return

            now = datetime.now(PT)
            if self.pst_start == "CLOSED" and self.pst_end == "CLOSED":
                liquidhours = 0
            elif now < self.pst_start or now > self.pst_end:
                liquidhours = 0
            else:
                liquidhours = 1

            imbalance = self.bid_size - self.ask_size
            timestamp = now.isoformat(sep=" ")

            self._strategy_on_tick(imbalance, self.last_price, liquidhours, now)

        # ------------------ Strategy State & Logic ------------------ #
        def _strategy_init(self):
            """
            Seed rolling averages from the last SLOW_WINDOW rows of imbalance_prices.csv.
            Called once after connection is ready and CSV is accessible.
            """
            df = pd.read_csv(
                CSV_FILE,
                usecols=["imbalance"]
            )
            # Take the last SLOW_WINDOW rows as seed
            seed = df.tail(SLOW_WINDOW)
            imbalances = seed["imbalance"].to_numpy(dtype=np.float64)
            n = len(imbalances)

            # Circular buffer holds the last SLOW_WINDOW values
            self._strat_buf = deque(imbalances.tolist(), maxlen=SLOW_WINDOW)

            # Running sums for O(1) updates
            sw = SLOW_WINDOW
            fw = FAST_WINDOW

            self._slow_sum = float(np.sum(imbalances[-sw:])) if n >= sw else float(np.sum(imbalances))
            fast_slice = imbalances[-fw:] if n >= fw else imbalances
            self._fast_sum = float(np.sum(fast_slice))
            self._fast_count = min(n, fw)   # only relevant during warmup (should always be fw after seed)

            # Compute initial MAs
            self._slow_ma = self._slow_sum / min(n, sw)
            self._fast_ma = self._fast_sum / self._fast_count if self._fast_count > 0 else 0.0

            # Strategy state
            self._prev_sign = int(np.sign(self._fast_ma - self._slow_ma))
            self._position  = None   # None | "LONG" | "SHORT"
            self._entry_price = None
            self._prev_liq  = 0      # previous tick's liquidhours value
            self._strat_ready = True
            print(f"✅ Strategy seeded with {n} rows. fast_MA={self._fast_ma:.4f} slow_MA={self._slow_ma:.4f} sign={self._prev_sign}")

        def _strategy_on_tick(self, imbalance: float, price: float, liquidhours: int, now: datetime):
            if not getattr(self, "_strat_ready", False):
                return

            sw = SLOW_WINDOW
            fw = FAST_WINDOW

            # --- Update circular buffer and running sums ---
            outgoing = self._strat_buf[0] if len(self._strat_buf) == sw else None
            self._strat_buf.append(imbalance)

            # Slow sum: add new, subtract the value that just left the window
            if outgoing is not None:
                self._slow_sum += imbalance - outgoing
            else:
                self._slow_sum += imbalance

            # Fast sum: the fast window is entirely inside the slow buffer
            # The fast window is the LAST fw elements of the buffer.
            # When a new element is appended, the element leaving the fast window
            # is the one that was at position (len_before - fw) = (sw-1 - fw) from the end before append,
            # which is now at position [sw - 1 - fw] in the buffer (0-indexed from oldest).
            buf_len = len(self._strat_buf)
            if buf_len > fw:
                # The element that just left the fast window
                fast_out = self._strat_buf[buf_len - fw - 1]
                self._fast_sum += imbalance - fast_out
            else:
                # Still filling fast window
                self._fast_sum += imbalance
                self._fast_count = buf_len

            fast_ma = self._fast_sum / fw if buf_len >= fw else self._fast_sum / buf_len
            slow_ma = self._slow_sum / min(buf_len, sw)

            curr_sign = 1 if fast_ma > slow_ma else (-1 if fast_ma < slow_ma else 0)
            timestamp_str = now.strftime("%Y-%m-%dT%H:%M:%S%z")

            prev_liq = self._prev_liq
            self._prev_liq = liquidhours

            # --- Market OPEN: first-of-day entry (prev_liq==0 → liq==1) ---
            if prev_liq == 0 and liquidhours == 1:
                side = "LONG" if fast_ma > slow_ma else "SHORT"
                self._position   = side
                self._entry_price = price
                pct = 0.0  # entry tick, no prior trade to compare
                print(f"{timestamp_str} {side} @ {price} ({pct:.4f}%)")
                self._log_trade(timestamp_str, side, price, pct)
                self._prev_sign = curr_sign
                return

            # --- Market CLOSE: force-close (liq==1 → prev was 1, now==0) ---
            if prev_liq == 1 and liquidhours == 0:
                if self._position is not None and self._entry_price is not None:
                    if self._position == "LONG":
                        pct = (price - self._entry_price) / self._entry_price * 100.0 - TRADE_FEE * 100.0
                    else:
                        pct = (self._entry_price - price) / self._entry_price * 100.0 - TRADE_FEE * 100.0
                    print(f"{timestamp_str} CLOSE @ {price} ({pct:.4f}%)")
                    self._log_trade(timestamp_str, "CLOSE", price, pct)
                self._position    = None
                self._entry_price = None
                self._prev_sign   = curr_sign
                return

            # --- Intraday crossover signals (only during liquid hours) ---
            if liquidhours == 1 and curr_sign != self._prev_sign and curr_sign != 0:
                side = "LONG" if curr_sign == 1 else "SHORT"

                # Compute pct return from previous position
                if self._position is not None and self._entry_price is not None:
                    if self._position == "LONG":
                        pct = (price - self._entry_price) / self._entry_price * 100.0 - TRADE_FEE * 100.0
                    else:
                        pct = (self._entry_price - price) / self._entry_price * 100.0 - TRADE_FEE * 100.0
                else:
                    pct = 0.0

                self._entry_price = price
                self._position    = side
                print(f"{timestamp_str} {side} @ {price} ({pct:.4f}%)")
                self._log_trade(timestamp_str, side, price, pct)

            self._prev_sign = curr_sign


        # ------------------ Connection & MarketData Subscription ------------------ #
        def safe_connect_and_run(self):
            print("Connecting to IB Gateway...")
            self.connect(self.host, self.port, clientId=self.client_id)
            threading.Thread(target=self.run, daemon=True).start()


        def nextValidId(self, orderId: int):
            super().nextValidId(orderId)
            if self._subscribed:
                return
            self._subscribed = True
            print("API ready, subscribing to TSLA...")
            self.reqMarketDataType(1)
            spy = Contract()
            spy.symbol = "TSLA"
            spy.secType = "STK"
            spy.exchange = "SMART"
            spy.primaryExchange = "NASDAQ"
            spy.currency = "USD"
            self.reqMktData(1, spy, "", False, False, [])
            self._strategy_init()


        # ------------------ Error Handler ------------------ #
        def error(self, *args):
            try:
                reqId = args[0] if len(args) > 0 else None
                errorCode = args[1] if len(args) > 1 else None
                errorMsg = args[2] if len(args) > 2 else None

                print(f"❗ ERROR -> reqId: {reqId}, code: {errorCode}, msg: {errorMsg}")

                if len(args) > 3:
                    print("Additional error info:", args[3:])

            except Exception as e:
                print("Error handler failed:", e)





    return IBApp


# ------------------ External Market Hours Fetch ------------------ #
def fetch_market_hours_from_external_script():
    try:
        result = subprocess.run(
            ["python", "market_hours_fetcher.py", "trader"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        if output:
            print("📡 Fetched liquidHours:", output)
            return output
        else:
            print("⚠️ No output from market_hours_fetcher.py")
            return None
    except subprocess.CalledProcessError as e:
        print("❌ Error running market_hours_fetcher.py:", e)
        return None


def parse_liquid_hours(raw_liquid_hours):
    today_str = datetime.now(ET).strftime("%Y%m%d")
    entries = [p for p in raw_liquid_hours.split(";") if p.startswith(today_str)]
    if not entries or "CLOSED" in entries[0]:
        return "CLOSED", "CLOSED"  # Market closed
    start_str, end_str = entries[0].split("-")
    start_dt = ET.localize(datetime.strptime(start_str, "%Y%m%d:%H%M"))
    end_dt = ET.localize(datetime.strptime(end_str, "%Y%m%d:%H%M")) - timedelta(seconds=1)
    pst_start = start_dt.astimezone(PT)
    pst_end = end_dt.astimezone(PT)
    return pst_start, pst_end

def wait_for_ib_api(host="127.0.0.1", port=4002, timeout=120):
    """
    Wait until IB Gateway/TWS API socket becomes available again.
    No sleeps guessing restart time—just real port check.
    """
    start = time.time()

    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                print("✅ IB API is available again")
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)

    print("❌ Timeout waiting for IB API")
    return False

def daily_hours_scheduler_external(app_holder):
    while True:
        now = datetime.now(PT)
        target = PT.localize(datetime.combine(now.date(), dtime(3, 5))) # 3, 5
        if now >= target:
            target += timedelta(days=1)
        time.sleep((target - now).total_seconds())
        print("🔄 Running external market hours fetch at 3:05 AM:")
        raw_hours = fetch_market_hours_from_external_script()
        if raw_hours:
            pst_start, pst_end = parse_liquid_hours(raw_hours)
            # always set on the current app
            app = app_holder['app']
            if app:
                app.pst_start = pst_start
                app.pst_end = pst_end
                app._hours_set = True
                print("🕒 Updated market hours PST:", pst_start, "→", pst_end)
                print()


def schedule_reconnect(app_holder, hour: int, minute: int):
    def loop():
        while True:
            now = datetime.now(PT)
            target = PT.localize(datetime.combine(now.date(), dtime(hour, minute)))

            if now >= target:
                target += timedelta(days=1)

            time.sleep((target - now).total_seconds())

            print("🔄 Scheduled reconnect triggered")

            app = app_holder.get("app")
            
            time.sleep(5)
            # 1. DISCONNECT immediately
            if app:
                try:
                    print("🔌 Disconnecting IB API...")
                    app.disconnect()
                except Exception as e:
                    print("Disconnect error:", e)

            # 2. WAIT until API is actually back
            time.sleep(5)
            print("⏳ Waiting for IB Gateway API to come back...")
            if not wait_for_ib_api("127.0.0.1", 4002, timeout=180):
                print("⚠️ Skipping reconnect (API not ready)")
                continue

            # 3. RECONNECT cleanly
            print("🔁 Reconnecting IB API...")

            IBApp = make_ib_app_instance()
            new_app = IBApp(
                host="127.0.0.1",
                port=4002,
                client_id=CLIENT_ID
            )

            # preserve state
            if app:
                new_app.pst_start = app.pst_start
                new_app.pst_end = app.pst_end
                new_app._hours_set = app._hours_set

            app_holder["app"] = new_app
            new_app.safe_connect_and_run()

    threading.Thread(target=loop, daemon=True).start()


# ------------------ Main ------------------ #
if __name__ == "__main__":
    # initial launch
    IBApp = make_ib_app_instance()
    app = IBApp(host="127.0.0.1", port=4002, client_id=CLIENT_ID)
    app_holder['app'] = app
    app.safe_connect_and_run()  # creates the network thread


    # Schedule daily 3:02 AM market hours fetch using external script
    threading.Thread(target=daily_hours_scheduler_external, args=(app_holder,), daemon=True).start()


    # Schedule IB Gateway automation at 3:02 AM PST
    schedule_reconnect(app_holder, 3, 2) # 3, 2


    # Keep script alive
    while True:
        time.sleep(1)
