from google import genai
from google.genai import types
from PIL import Image
from ultralytics import YOLO
import io as _io
import os
import pickle
import re
import asyncio
import difflib
import json 
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timedelta
import sqlite3
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters as tg_filters
from functools import wraps
import numpy as np
PERSON_AND_BAGS = [0, 24, 25, 26, 28]
load_dotenv()
# GEMINI_KEY = os.getenv('GEMINI_API_KEY')
# _client = None
# if os.getenv('GEMINI_API_KEY'):
#     _client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
_yolo = YOLO('yolov8m.pt')
TOKEN = os.getenv('TELEGRAM_TOKEN')

with open('models/crowd_model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

TABLE = MODEL['table']
STATIONS = MODEL['stations']
LIVE = {'grams': None, 'ts': 0.0, 'source': None}
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
        n = int(self.headers.get('Content-Length', 0))

        if self.path == '/image':
            img_bytes = self.rfile.read(n)
            try:
                content_types = self.headers.get('Content-Type', '')
                if 'octet-stream' in content_types:
                    w = int(self.headers.get('X-Frame-Width', 320))
                    h = int(self.headers.get('X-Frame-Height', 240))
                    arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape((h, w, 2))
                    r = (arr[:,:,1] & 0xF8)
                    g = ((arr[:,:,1] & 0x07) << 5) | ((arr[:,:,0] & 0xE0) >> 3)
                    b = (arr[:,:,0] & 0x1F) << 3
                    img = Image.fromarray(np.stack([r,g,b], axis=2).astype(np.uint8))
                else:
                    img = Image.open(_io.BytesIO(img_bytes)).convert('RGB')
                results = _yolo(
                    np.array(img), 
                    verbose=False,
                    classes=PERSON_AND_BAGS,
                    conf=0.2,
                    iou=0.4
                )
                count = sum(
                    1 for box in results[0].boxes
                    if int(box.cls) == 0 and float(box.conf) > 0.4
                )
                LIVE['grams'] = count * (FULL / 5)
                LIVE['ts'] = time.time()
                LIVE['source'] = 'camera'
                log_reading(LIVE['grams'])
                print(f'[CAM] {count} people -> {LIVE['grams']:.0f}g')
                self.send_response(200)
            except Exception as e:
                print(f'[CAM] YOLO failed: {e}')
                self.send_response(500)
            self.end_headers()

        elif self.path == '/weight':
            try:
                data = json.loads(self.rfile.read(n))
                LIVE['grams']  = float(data['grams'])
                LIVE['ts']     = time.time()
                LIVE['source'] = 'weight'
                log_reading(LIVE['grams'])
                print(f'[WEIGHT] {LIVE["grams"]:.0f}g')
                self.send_response(200)
            except Exception as e:
                print(f'[WEIGHT] Parse failed: {e}')
                self.send_response(400)

        else:
            self.send_response(404)

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
    pct    = max(0.0, min(1.0, LIVE['grams'] / FULL))
    source = 'camera' if LIVE['source'] == 'camera' else 'weight sensor'
    return (f"Prototype coach (live): {pct:.0%} ({level(pct)}) "
            f"| {LIVE['grams']:.0f}g on platform "
            f"| via {source}")
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
        "/photo  send a photo to count people or figures in it\n"
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
        pct = max(0.0, min(1.0, LIVE['grams'] / FULL))
        proto = (f"Prototype: online\n"
                 f"Last reading: {LIVE['grams']:.0f} g ({pct:.0%})\n"
                 f"{age:.0f} s ago")
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT COUNT(*), MIN(ts), MAX(ts) FROM readings").fetchone()
    con.close()
    db = (f"DB: {row[0]} readings\nFirst: {row[1]}\nLast: {row[2]}"
          if row[0] else "DB: empty")
    await update.message.reply_text(f"{proto}\n\n{db}")
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
    lines.append(f"{'Day type':<12} {'Avg':>6} {'Peak':>6} {'Reading':>9}")
    lines.append('-'*40)
    for day_type, row in summary.iterrows():
        if pd.isna(row['readings']) or row['readings'] == 0:
            lines.append(f"{day_type:<12} {'--':>6} {'--':>6}{'0':>9}")
        else:
            lines.append(
                f"{day_type:<12} {row['avg_load']:>6} {row['peak_load']:>6} {int(row['readings']):>9}")
    lines[0] = 'Prototype stats (Pattanagere coach)\n'
    lines.append(f'\nTotal readings: {len(df)}')
    ts_min = df['ts'].min().date()
    ts_max = df['ts'].max().date()
    lines.append(f'Period: {ts_min} to {ts_max}')
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

async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.photo:
        await update.message.reply_text("Send a photo of the coach")
        return

    await update.message.reply_text('Analysing...')

    file = await context.bot.get_file(update.message.photo[-1].file_id)
    buf = bytes(await file.download_as_bytearray())

    def run_yolo():
        img = Image.open(_io.BytesIO(buf)).convert('RGB')
        results = _yolo(np.array(img), verbose=False, classes=PERSON_AND_BAGS)
        person_count = sum(
            1 for box in results[0].boxes
            if int(box.cls) == 0 and float(box.conf) > 0.10
        )
        bag_count = sum(
            1 for box in results[0].boxes
            if int(box.cls) in [24, 25, 26, 28] and float(box.conf) > 0.10
        )
        annotated = results[0].plot()
        annotated_pill = Image.fromarray(annotated)
        out = _io.BytesIO()
        annotated_pill.save(out, format='JPEG', quality=85)
        out.seek(0)
        return person_count, bag_count, out

    try:
        count, bags,annotated_img  = await asyncio.to_thread(run_yolo)
        pct = min(100, round(count / 15 * 100))
        lbl = level(pct / 100)

        await update.message.reply_photo(
            photo=annotated_img,
            caption=(
                f'Vision Count: {count} passenger(s) detected\n'
                f'Bags/Luggage: {bags} detected\n'
                f'Estimated coach load: {pct}% ({lbl})\n\n'
                f'(by YOLOv8, max capacity = 15 for this prototype!)')
            )
    except Exception as e:
        await update.message.reply_text(f'Detection failed: {e}')

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
    app.add_handler(CommandHandler('photo', photo))
    app.add_handler(MessageHandler(tg_filters.PHOTO, photo))
    print('Bot running. Ctrl + C to kill. hehehehehe')
    init_db()
    start_receiver()
    app.run_polling()

if __name__ == '__main__':
    main()
                                    