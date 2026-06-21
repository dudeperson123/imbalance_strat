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


# ------------------ Config ------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "data", "imbalance_prices.csv")
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)


PT = pytz.timezone("US/Pacific")
ET = pytz.timezone("US/Eastern")


# For holding the currently active IBApp (for threading compatibility)
app_holder = {'app': None}

CLIENT_ID = 1

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


            # CSV setup
            if not os.path.exists(CSV_FILE):
                with open(CSV_FILE, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["now", "imbalance", "price", "liquidhours", "bid_size", "ask_size"])


        # ------------------ Market Data ------------------ #
        def tickSize(self, tickerId, field, size):
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
            with open(CSV_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, imbalance, self.last_price, liquidhours, self.bid_size, self.ask_size])


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
            ["python", "market_hours_fetcher.py"],
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


            # 2. WAIT until API thread fully shuts down
            if app and hasattr(app, "_run_thread"):
                print("⏳ Waiting for IB API thread to terminate...")
                app._run_thread.join(timeout=5)

                if app._run_thread.is_alive():
                    print("⚠️ IB API thread still alive after timeout. Possibility of trivial missed ticks.")

            # 3. WAIT until API is actually back
            print("⏳ Waiting for IB Gateway API to come back...")
            if not wait_for_ib_api("127.0.0.1", 4002, timeout=180):
                print("⚠️ Skipping reconnect (API not ready)")
                continue

            # 4. RECONNECT cleanly
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
