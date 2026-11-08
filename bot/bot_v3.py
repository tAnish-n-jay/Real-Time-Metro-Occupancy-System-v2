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
import io
from PIL import Image
import google.generativeai as genai
from telegram.ext import Application, CommandHandler, ContextTypes
from functools import wraps
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

with open('models/crowd_model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

TABLE = MODEL['table']
STATIONS = MODEL['stations']
LIVE = {'grams': None, 'ts': 0.0}

LIVE_VISION = {'count': None, 'ts':0.0}
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
GEMINI = genai.GenerativeModel('gemini-2.5-flash')
FULL = 1500.0
DEMO_STATION = 'Pattanagere'
ADMIN_IDS = set(
    int(x.strip())
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x.strip()
)

def admin_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id not in ADMIN_IDS:
            await update.message.reply_text("You are not authorised to use this command.")
            return
        return await func(update, context)
    return wrapper
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
        path = self.path.split('?')[0]

        if path == '/upload':
            try:
                n = int(self.headers.get('Content-Length', 0))
                img_bytes = self.rfile.read(n)
                image = Image.open(io.BytesIO(img_bytes))
                resp = GEMINI.generate_content([
                    'Count the number of people visible in this image. '
                    'Reply with just a number. If none, reply 0.',
                    image
                ])
                count = int(resp.text.stip())
            except Exception as e:
                print(f'error: {e}')
                self.send_response(400); self.end_headers(); return
            LIVE_VISION['count'] = count
            LIVE_VISION['ts'] = time.time()
            print(f'[vision] count: {count} people')
            self.send_response(200); self.end_headers()
        else:
            try:
                n = int(self.headers.get('Content-Length', 0))
                data = json.loads(self.rfile.read(n))
                LIVE['grams'] = float(data['grams'])
                LIVE['ts'] = time.time()
                self.send_response(200)
            except Exception:
                self.send_response(400)
                self.end_headers()
                return
            try:
                log_reading(LIVE['grams'])
            except Exception as e:
                print('DB log failed -', e)
            self.end_headers()

    def log_message(self, *args):
        pass

def start_receiver():
    server = HTTPServer(('0.0.0.0', PORT), Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'Receiver listening on port {PORT}')

def live_text() -> str:
    lines = []

    if LIVE['grams'] is None or time.time() - LIVE['ts'] > STALE_SECOND:
        lines.append('Prototype coach: offline (no recent reading).')
    else:
        pct = max(0.0, min(1.0, LIVE["grams"] / FULL))
        lines.append(f"Prototype coach (live): {pct:.0%} ({level(pct)}), {LIVE['grams']:.0f} g on the platform")

    if LIVE_VISION['count'] is not None and time.time() - LIVE_VISION['ts'] < 300:
        age = int(time.time() - LIVE_VISION['ts'])
        lines.append(f'Camera count: {LIVE_VISION['count']} people ({age}s ago)')
    else:
        lines.append('Camera: no recent reading')

    return '\n'.join(lines)
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
        "/crowd <station> <day> <time>  e.g. /crowd majestic mon 9am\n"
        "/now <station>  live prototype reading + typical crowd\n"
        "/whoami  your Telegram user ID\n\n"
        "Admin only:\n"
        "/status  prototype health and DB info\n"
        "/stats  load summary by day type\n"
        "/history  chart of the last 60 readings\n\n"
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
async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    await update.message.reply_text(
        f"Your Telegram user ID: `{u.id}`\nName: {u.full_name}",
        parse_mode="Markdown"
    )

@admin_only
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    age = time.time() - LIVE['ts']
    if LIVE['grams'] is None or age > STALE_SECOND:
        proto = 'Prototype: offline'
    else:
        pct = max(0,0, min(1.0, LIVE['grams'] / FULL))
        proto = (f"Prototype: online\n",
                 f"Last reading: {LIVE['grams']:.0f} f ({pct:.0%})\n"
                 f"{age:.0f} s ago")
        con = sqlite3.connect(DB_PATH)
        row = con.execute(
            "SELECT COUNT(*), MIN(ts), MAX(ts) FROM readings").fetchone()
        con.close()
        db = (f"DB: {row[0]} readings\n"
              f"First: {row[1]}\nLast: {row[2]}" if row[0] else "DB: empty")
        await update.message.reply_text(f'{proto}\n\n{db}')
@admin_only
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT ts, grams, percent FROM readings ORDER BY ts"
    ).fetchall()
    con.close()
    if not rows:
        await update.message.reply_text("No readings in the database yet.")
        return
    import pandas as pd
    df = pd.DataFrame(rows, columns=['ts', 'grams', 'percent'])
    df['ts'] = pd.to_datetime(df['ts'])
    df['hour'] = df['ts'].dt.hour
    df['dow'] = df['ts'].dt.day_name()
    df['day_type'] = df['ts'].dt.day_of_week.apply(
        lambda d: "Weekday" if d < 5 else ("Saturday" if d == 5 else "Sunday"))
    summary = df.groupby("day_type").agg(
        avg_load=('percent', lambda x: f"{x.mean():.0%}"),
        peak_load=('percent', lambda x: f"{x.max():.0%}"),
        readings=("percent","count")
    ).reindex(['Weekday', 'Saturday', 'Sunday'])
    lines = ['**Prototype stats (pattanagere coach)*\n']
    lines.append(f'{'Day type':<12} {'Avg':>6} {'Peak':>6} {'Reading':>9}')
    lines.append('-'*40)
    for day_type, row in summary.iterrows():
        if pd.isna(row['readings']) or row['readings'] == 0:
            lines.append(f'{day_type:<12} {'--':>6} {'--':>6}{'0':>9}')
        else:
            lines.append(
                f"{day_type:<12} {row['avg_load']:>6} {row['peak_load']:>6} {int(row['readings']):>9}")
    lines[0] = 'Prototype stats (Pattanagere coach)\n'
    lines.append(f'\nTotal readings: {len(df)}')
    lines.append(f'Period: {df['ts'].min().date()} to {df['ts'].max().date()}')
    await update.message.reply_text("\n".join(lines))

@admin_only
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT ts, grams, percent FROM readings ORDER BY ts DESC LIMIT 60"
    ).fetchall()
    con.close()

    if not rows:
        await update.message.reply_text("No readings yet.")
        return

    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io

    df = pd.DataFrame(rows[::-1], columns=["ts", "grams", "percent"])
    df["ts"] = pd.to_datetime(df["ts"])
    df["pct"] = df["percent"] * 100

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df['ts'], df['pct'], color='#5b8dee', linewidth=1.8)
    ax.fill_between(df['ts'], df['pct'], alpha=0.15, color='#5b8dee')
    ax.axhline(100, color='red', linestyle='--', linewidth=0.8, label= "Full (1500 g)")
    ax.set_ylim(0, 110)
    ax.set_ylabel('Coach load (%)')
    ax.set_xlabel('Time')
    ax.set_title("Pattanagere prototype - last 60 readings")
    ax.legend(fontsize=8)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    plt.close(fig)

    await update.message.reply_photo(buf, caption='Last 60 readings from the prototype coach.')

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_cmd))
    app.add_handler(CommandHandler('crowd', crowd))
    app.add_handler(CommandHandler('now', now))
    app.add_handler(CommandHandler('whoami', whoami))
    app.add_handler(CommandHandler('status', status))
    app.add_handler(CommandHandler('stats', stats))
    app.add_handler(CommandHandler('history', history))
    print('Bot running. Ctrl + C to kill. hehehehehe')
    init_db()
    start_receiver()
    app.run_polling()

if __name__ == '__main__':
    main()
                                    