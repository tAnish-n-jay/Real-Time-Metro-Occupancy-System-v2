import os
import pickle
import re
import difflib
import json 
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timedelta
import sqlite3
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

with open('models/crowd_model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

TABLE = MODEL['table']
STATIONS = MODEL['stations']
LIVE = {'grams': None, 'ts': 0.0}
FULL = 1500.0
DEMO_STATION = 'Pattanagere'
DB_PATH = 'data/live.db'


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS readings(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        grams REAL NOT NULL,
        percent REAL NOT NULL)""")
def log_reading(grams: float):
    pct = max(0.0, min(1.0, grams / FULL))
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute('INSERT INTO readings (ts, grams, percent) VALUES (?, ?, ?)',
                    (datetime.now().isoformat(timespec='seconds'), grams, pct))
        con.commit()
    finally:
        con.close()


STALE_SECOND = 15
PORT = 5000

class Receiver(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            n = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(n))
            LIVE["grams"] = float(data['grams'])
            LIVE['ts'] = time.time()
            self.send_response(200)
        except Exception:
            self.send_response(400)
            self.end_headers()
            return
        try:
            log_reading(LIVE['grams'])
        except Exception as e:
            print('DB log failed - ', e)
        self.end_headers()

    def log_message(self, *args):
        pass

def start_receiver():
    server = HTTPServer(('0.0.0.0', PORT), Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'Receiver listening on port {PORT}')

def live_text() -> str:
    if LIVE['grams'] is None or time.time() - LIVE['ts'] > STALE_SECOND:
        return 'Prototype coach: offline (no recent reading).'
    pct = max(0.0, min(1.0, LIVE["grams"] / FULL))
    return f"Prototype coach (live): {pct:.0%} ({level(pct)}), {LIVE['grams']:.0f} g on the platform"
DAYS = {
    'mon': ('weekday', 'Monday'), 'monday': ('weekday', 'Monday'),
    "tue": ("weekday", "Tuesday"), "tues": ("weekday", "Tuesday"), "tuesday": ("weekday", "Tuesday"),
    "wed": ("weekday", "Wednesday"), "wednesday": ("weekday", "Wednesday"),
    "thu": ("weekday", "Thursday"), "thur": ("weekday", "Thursday"), "thursday": ("weekday", "Thursday"),
    "fri": ("weekday", "Friday"), "friday": ("weekday", "Friday"),
    "weekday": ("weekday", "Weekday"), "weekdays": ("weekday", "Weekday"),
    "sat": ("saturday", "Saturday"), "saturday": ("saturday", "Saturday"),
    "sun": ("sunday", "Sunday"), "sunday": ("sunday", "Sunday"),
}

def day_type_of(dt: datetime) -> str:
    return 'weekday' if dt.weekday() < 5 else ('saturday' if dt.weekday() == 5 else 'sunday')

def parse_day(word: str):
    w = word.lower()
    if w in DAYS:
        return DAYS[w]
    if w == 'today':
        d = datetime.now()
        return day_type_of(d), 'Today (' + d.strftime('%A') + ')'
    if w == 'tomorrow':
        d = datetime.now() + timedelta(days=1)
        return day_type_of(d), 'Tomorrow (' + d.strftime('%A') + ')'
    return None

def parse_hour(text: str):
    m = re.fullmatch(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text.lower().strip())
    if not m:
        return None
    h, ap = int(m.group(1)), m.group(3)
    if ap:
        if not 1 <= h <= 12:
            return None
        h = h % 12 + (12 if ap == 'pm' else 0)
    return h if 0 <= h <= 23 else None

def find_station(query: str):
    """Return a list of matching station names(empty if none)"""

    q = query.lower().strip()
    if not q:
        return []
    lower = {s.lower(): s for s in STATIONS}
    if q in lower:
        return [lower[q]]
    partial = [s for s in STATIONS if q in s.lower()]
    if partial:
        return partial
    pieces = {}
    for s in STATIONS:
        for p in [s.lower()] + [x.strip() for x in s.lower().split(',')]:
            pieces.setdefault(p, s)
    out = []
    for c in difflib.get_close_matches(q, list(pieces), n=5, cutoff=0.6):
        if pieces[c] not in out:
            out.append(pieces[c])
    return out

def level(c: float) -> str:
    return 'Low' if c < 0.25 else 'Moderate' if c < 0.5 else 'Crowded' if c < 0.75 else 'Very Crowded'

def hour_str(h: int) -> str:
    start = datetime(2000, 1, 1, h)
    end = start.replace(minute=59)
    return f"{start.strftime('%I:%M %p').lstrip('0')} to {end.strftime('%I:%M %p').lstrip('0')}"

def answer(station_query: str, day_type: str, day_label: str, hour: int) -> str:
    matches = find_station(station_query)
    if not matches:
        return f"Station {station_query} not found! Check spelling."
    if len(matches) > 1:
        names = "\n".join('-' + m for m in matches[:5])
        return f"'{station_query}' matches several stations. Please be more specific:\n{names}"
    st = matches[0]
    c = TABLE[(st, day_type, hour)]
    note = '\n(little to no ridership at this hour)' if c == 0 else ""
    return (f'{st}\n{day_label}, {hour_str(hour)}\n'
            f"Crowd level: {c:.0%} ({level(c)}) {note}\n\n"
            "Predicted from historical data, relative to this station's busy level. It is not based on the live count.")

HELP = ("Commands:\n"
        "/crowd <station> <day> <time> e.g. /crowd majestic mon 9am\n"
        "/now <station> live prototype reading + typical crowd for this day and hour\n"
        "Days: mon..sun, weekday, today, tomorrow\n"
        "Time: 9, 9am, 6pm, 23:00")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Namma Metro Crowd Predictor.\n\n' + HELP)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)

async def crowd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 3:
        await update.message.reply_text('Usage: /crowd <station> <day> <time>\nExample: /crowd majestic mon 9am')
        return 
    hour, day = parse_hour(args[-1]), parse_day(args[-2])
    if hour is None:
        await update.message.reply_text(f"Try again! '{args[-1]}'. Try 9, 9am, 9pm or 21:00.")
        return 
    if day is None:
        await update.message.reply_text(f'Try again! "{args[-2]}". Try mon..sun, weekday, today or tomorrow.')
        return
    await update.message.reply_text(answer(" ".join(args[:-2]), day[0], day[1], hour))

async def now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d = datetime.now()
    if not context.args:
        await update.message.reply_text(
            live_text() + "\nAdd a station to see the predicted crowd, e.g, /now pattanagere.")
        return 
    query = " ".join(context.args)
    matches = find_station(query)
    parts = []
    if len(matches) == 1 and matches[0] == DEMO_STATION:
        parts.append(live_text())
    parts.append(answer(query, day_type_of(d), d.strftime('%A'), d.hour))
    if len(matches) == 1 and matches[0] != DEMO_STATION:
        parts.append(f'(Live prototype coach is only at {DEMO_STATION.split(',')[-1].strip()}.)')
    await update.message.reply_text('\n\n'.join(parts))
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_cmd))
    app.add_handler(CommandHandler('crowd', crowd))
    app.add_handler(CommandHandler('now', now))
    print('Bot running. Ctrl + C to kill. hehehehehe')
    init_db()
    start_receiver()
    app.run_polling()

if __name__ == '__main__':
    main()
                                    