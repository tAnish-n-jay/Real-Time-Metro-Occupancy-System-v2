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

load_dotenv()
TOKEN     = os.getenv('TELEGRAM_TOKEN')
DB_PATH   = 'data/live.db'
PORT      = 5000
FULL_G    = 1500.0   
AVG_PERSON_G = 65.0  
MAX_PERSONS  = 15    
DEMO_STATION = 'Pattanagere'
STALE_SECOND = 30
PERSON_AND_BAGS = [0, 24, 25, 26, 28]  

ADMIN_IDS = set(
    int(x.strip())
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x.strip()
)

print('[YOLO] Loading models...')
_yolo_models = {
    'nano'  : YOLO('models/yolov8n.pt'),  # < 25% 
    'small' : YOLO('models/yolov8s.pt'),  # 25-50%
    'medium': YOLO('models/yolov8m.pt'),  # 50-75%
    'large' : YOLO('models/yolov8l.pt'),  # > 75% 
}
print('[YOLO] All models ready.')

def pick_model(grams: float) -> tuple:
    """Pick YOLO model based on current weight sensor reading."""
    pct = (grams / FULL_G) if grams else 0.0
    if pct < 0.25:
        return _yolo_models['nano'],   'YOLOv8n (Low crowd)'
    elif pct < 0.50:
        return _yolo_models['small'],  'YOLOv8s (Moderate crowd)'
    elif pct < 0.75:
        return _yolo_models['medium'], 'YOLOv8m (Crowded)'
    else:
        return _yolo_models['large'],  'YOLOv8l (Very Crowded)'

with open('models/crowd_model.pkl', 'rb') as f:
    MODEL = pickle.load(f)
TABLE    = MODEL['table']
STATIONS = MODEL['stations']

LIVE = {'grams': None, 'ts': 0.0, 'source': None}

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS readings(
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      TEXT    NOT NULL,
            grams   REAL    NOT NULL,
            percent REAL    NOT NULL)""")

def log_reading(grams: float):
    pct = max(0.0, min(1.0, grams / FULL_G))
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute(
            'INSERT INTO readings (ts, grams, percent) VALUES (?, ?, ?)',
            (datetime.now().isoformat(timespec='seconds'), grams, pct))
        con.commit()
    finally:
        con.close()

def combined_occupancy(grams: float, person_count: int) -> dict:
    """Fuse weight sensor + camera for best occupancy estimate."""
    weight_pct    = max(0.0, min(1.0, grams / FULL_G))
    weight_persons = round(grams / AVG_PERSON_G)
    camera_pct    = max(0.0, min(1.0, person_count / MAX_PERSONS))

    if grams > 50: 
        combined_pct = (weight_pct * 0.6) + (camera_pct * 0.4)
    else:          
        combined_pct = camera_pct

    return {
        'combined_pct'   : combined_pct,
        'weight_pct'     : weight_pct,
        'camera_pct'     : camera_pct,
        'weight_persons' : weight_persons,
        'camera_persons' : person_count,
        'estimated_count': max(weight_persons, person_count),
    }

class Receiver(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))

        if self.path == '/image':
            img_bytes = self.rfile.read(n)
            try:
                content_type = self.headers.get('Content-Type', '')
                if 'octet-stream' in content_type:
                    w   = int(self.headers.get('X-Frame-Width',  320))
                    h   = int(self.headers.get('X-Frame-Height', 240))
                    arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape((h, w, 2))
                    r   = (arr[:,:,1] & 0xF8)
                    g   = ((arr[:,:,1] & 0x07) << 5) | ((arr[:,:,0] & 0xE0) >> 3)
                    b   = (arr[:,:,0] & 0x1F) << 3
                    img = Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8))
                else:
                    img = Image.open(_io.BytesIO(img_bytes)).convert('RGB')

                grams        = LIVE['grams'] or 0.0
                model, mname = pick_model(grams)

                results = model(
                    np.array(img),
                    verbose=False,
                    classes=PERSON_AND_BAGS,
                    conf=0.2,
                    iou=0.4
                )
                person_count = sum(
                    1 for box in results[0].boxes
                    if int(box.cls) == 0
                )
                bag_count = sum(
                    1 for box in results[0].boxes
                    if int(box.cls) in [24, 25, 26, 28]
                )

                occ = combined_occupancy(grams, person_count)

                LIVE['grams']  = occ['combined_pct'] * FULL_G
                LIVE['ts']     = time.time()
                LIVE['source'] = 'combined' if grams > 50 else 'camera'
                log_reading(LIVE['grams'])

                print(f'[CAM] {person_count} people, {bag_count} bags '
                      f'| weight={grams:.0f}g '
                      f'| combined={occ["combined_pct"]:.0%} '
                      f'| model={mname}')
                self.send_response(200)

            except Exception as e:
                print(f'[CAM] YOLO failed: {e}')
                self.send_response(500)
            self.end_headers()

        elif self.path == '/weight':
            try:
                data           = json.loads(self.rfile.read(n))
                LIVE['grams']  = float(data['grams'])
                LIVE['ts']     = time.time()
                LIVE['source'] = 'weight'
                log_reading(LIVE['grams'])
                print(f'[WEIGHT] {LIVE["grams"]:.0f}g')
                self.send_response(200)
            except Exception as e:
                print(f'[WEIGHT] Parse failed: {e}')
                self.send_response(400)
            self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass

def start_receiver():
    server = HTTPServer(('0.0.0.0', PORT), Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'[HTTP] Receiver listening on port {PORT}')

def level(c: float) -> str:
    return 'Low' if c < 0.25 else 'Moderate' if c < 0.5 else 'Crowded' if c < 0.75 else 'Very Crowded'

def live_text() -> str:
    if LIVE['grams'] is None or time.time() - LIVE['ts'] > STALE_SECOND:
        return 'Prototype coach: offline (no recent reading).'
    pct    = max(0.0, min(1.0, LIVE['grams'] / FULL_G))
    source = {'camera': '📷 camera', 'weight': '⚖️ weight sensor', 'combined': '🔀 combined'}.get(
        LIVE['source'], LIVE['source'])
    return (f"Prototype coach (live): {pct:.0%} ({level(pct)}) "
            f"| {LIVE['grams']:.0f}g | via {source}")

def hour_str(h: int) -> str:
    start = datetime(2000, 1, 1, h)
    end   = start.replace(minute=59)
    return f"{start.strftime('%I:%M %p').lstrip('0')} to {end.strftime('%I:%M %p').lstrip('0')}"

def admin_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id not in ADMIN_IDS:
            await update.message.reply_text("You are not authorised to use this command.")
            return
        return await func(update, context)
    return wrapper

DAYS = {
    'mon': ('weekday', 'Monday'), 'monday': ('weekday', 'Monday'),
    'tue': ('weekday', 'Tuesday'), 'tues': ('weekday', 'Tuesday'), 'tuesday': ('weekday', 'Tuesday'),
    'wed': ('weekday', 'Wednesday'), 'wednesday': ('weekday', 'Wednesday'),
    'thu': ('weekday', 'Thursday'), 'thur': ('weekday', 'Thursday'), 'thursday': ('weekday', 'Thursday'),
    'fri': ('weekday', 'Friday'), 'friday': ('weekday', 'Friday'),
    'weekday': ('weekday', 'Weekday'), 'weekdays': ('weekday', 'Weekday'),
    'sat': ('saturday', 'Saturday'), 'saturday': ('saturday', 'Saturday'),
    'sun': ('sunday', 'Sunday'), 'sunday': ('sunday', 'Sunday'),
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

def answer(station_query: str, day_type: str, day_label: str, hour: int) -> str:
    matches = find_station(station_query)
    if not matches:
        return f"Station '{station_query}' not found! Check spelling."
    if len(matches) > 1:
        names = "\n".join('- ' + m for m in matches[:5])
        return f"'{station_query}' matches several stations:\n{names}"
    st = matches[0]
    c  = TABLE[(st, day_type, hour)]
    note = '\n(little to no ridership at this hour)' if c == 0 else ''
    return (f'{st}\n{day_label}, {hour_str(hour)}\n'
            f'Crowd level: {c:.0%} ({level(c)}){note}\n\n'
            "Predicted from historical data. Not based on live count.")

HELP = ("Commands:\n"
        "/crowd <station> <day> <time>  e.g. /crowd majestic mon 9am\n"
        "/now <station>  live prototype reading + typical crowd\n"
        "/whoami  your Telegram user ID\n\n"
        "Just send a photo to count people and estimate load!\n\n"
        "Admin only:\n"
        "/status  prototype health and DB info\n"
        "/stats   load summary by day type\n"
        "/history chart of last 60 readings\n\n"
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
        await update.message.reply_text(f"Bad time: '{args[-1]}'. Try 9, 9am, 9pm or 21:00.")
        return
    if day is None:
        await update.message.reply_text(f"Bad day: '{args[-2]}'. Try mon..sun, weekday, today or tomorrow.")
        return
    await update.message.reply_text(answer(" ".join(args[:-2]), day[0], day[1], hour))

async def now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    d = datetime.now()
    if not context.args:
        await update.message.reply_text(
            live_text() + "\nAdd a station to see predicted crowd, e.g. /now pattanagere.")
        return
    query   = " ".join(context.args)
    matches = find_station(query)
    parts   = []
    if len(matches) == 1 and matches[0] == DEMO_STATION:
        parts.append(live_text())
    parts.append(answer(query, day_type_of(d), d.strftime('%A'), d.hour))
    if len(matches) == 1 and matches[0] != DEMO_STATION:
        parts.append(f"(Live prototype coach is only at {DEMO_STATION.split(',')[-1].strip()}.)")
    await update.message.reply_text('\n\n'.join(parts))

async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    await update.message.reply_text(
        f"Your Telegram user ID: `{u.id}`\nName: {u.full_name}",
        parse_mode="Markdown")

@admin_only
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    age = time.time() - LIVE['ts']
    if LIVE['grams'] is None or age > STALE_SECOND:
        proto = 'Prototype: offline'
    else:
        pct   = max(0.0, min(1.0, LIVE['grams'] / FULL_G))
        proto = (f"Prototype: online\n"
                 f"Last reading: {LIVE['grams']:.0f}g ({pct:.0%})\n"
                 f"{age:.0f}s ago | via {LIVE['source']}")
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM readings").fetchone()
    con.close()
    db = (f"DB: {row[0]} readings\nFirst: {row[1]}\nLast: {row[2]}"
          if row[0] else "DB: empty")
    await update.message.reply_text(f"{proto}\n\n{db}")

@admin_only
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT ts, grams, percent FROM readings ORDER BY ts").fetchall()
    con.close()
    if not rows:
        await update.message.reply_text("No readings in the database yet.")
        return
    import pandas as pd
    df = pd.DataFrame(rows, columns=['ts', 'grams', 'percent'])
    df['ts']       = pd.to_datetime(df['ts'])
    df['day_type'] = df['ts'].dt.day_of_week.apply(
        lambda d: "Weekday" if d < 5 else ("Saturday" if d == 5 else "Sunday"))
    summary = df.groupby("day_type").agg(
        avg_load  =('percent', lambda x: f"{x.mean():.0%}"),
        peak_load =('percent', lambda x: f"{x.max():.0%}"),
        readings  =('percent', 'count')
    ).reindex(['Weekday', 'Saturday', 'Sunday'])
    lines = ['Prototype stats (Pattanagere coach)\n']
    lines.append(f"{'Day type':<12} {'Avg':>6} {'Peak':>6} {'Readings':>9}")
    lines.append('-' * 40)
    for day_type, row in summary.iterrows():
        if pd.isna(row['readings']) or row['readings'] == 0:
            lines.append(f"{day_type:<12} {'--':>6} {'--':>6} {'0':>9}")
        else:
            lines.append(
                f"{day_type:<12} {row['avg_load']:>6} {row['peak_load']:>6} {int(row['readings']):>9}")
    lines.append(f'\nTotal readings: {len(df)}')
    lines.append(f"Period: {df['ts'].min().date()} to {df['ts'].max().date()}")
    await update.message.reply_text("\n".join(lines))

@admin_only
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    con  = sqlite3.connect(DB_PATH)
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
    df["ts"]  = pd.to_datetime(df["ts"])
    df["pct"] = df["percent"] * 100

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(df['ts'], df['pct'], color='#5b8dee', linewidth=1.8)
    ax.fill_between(df['ts'], df['pct'], alpha=0.15, color='#5b8dee')
    ax.axhline(100, color='red', linestyle='--', linewidth=0.8, label="Full (1500g)")
    ax.set_ylim(0, 110)
    ax.set_ylabel('Coach load (%)')
    ax.set_xlabel('Time')
    ax.set_title("Pattanagere prototype — last 60 readings")
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
        await update.message.reply_text("Send a photo of the coach.")
        return

    await update.message.reply_text('Analysing...')

    file = await context.bot.get_file(update.message.photo[-1].file_id)
    buf  = bytes(await file.download_as_bytearray())

    grams = LIVE['grams'] or 0.0

    def run_yolo():
        model, model_name = pick_model(grams)
        img     = Image.open(_io.BytesIO(buf)).convert('RGB')
        results = model(
            np.array(img),
            verbose=False,
            classes=PERSON_AND_BAGS,
            conf=0.2,
            iou=0.4
        )
        person_count = sum(
            1 for box in results[0].boxes
            if int(box.cls) == 0
        )
        bag_count = sum(
            1 for box in results[0].boxes
            if int(box.cls) in [24, 25, 26, 28]
        )
        annotated = results[0].plot()
        out = _io.BytesIO()
        Image.fromarray(annotated).save(out, format='JPEG', quality=85)
        out.seek(0)
        return person_count, bag_count, out, model_name

    try:
        person_count, bag_count, annotated_img, model_name = await asyncio.to_thread(run_yolo)

        # Fuse camera + weight sensor
        occ = combined_occupancy(grams, person_count)
        lbl = level(occ['combined_pct'])

        # Weight sensor status line
        if LIVE['grams'] is None or time.time() - LIVE['ts'] > STALE_SECOND:
            weight_line = 'Weight sensor: offline'
        else:
            weight_line = f"Weight sensor: {grams:.0f}g ({occ['weight_pct']:.0%})"

        await update.message.reply_photo(
            photo=annotated_img,
            caption=(
                f"Passengers: {person_count} detected\n"
                f"Bags/Luggage: {bag_count} detected\n"
                f"{weight_line}\n"
                f"Combined load: {occ['combined_pct']:.0%} ({lbl})\n"
                f"Estimated count: {occ['estimated_count']} / {MAX_PERSONS}\n"
                f"Model: {model_name}"
            )
        )

        LIVE['grams']  = occ['combined_pct'] * FULL_G
        LIVE['ts']     = time.time()
        LIVE['source'] = 'combined' if grams > 50 else 'camera'
        log_reading(LIVE['grams'])

    except Exception as e:
        await update.message.reply_text(f'Detection failed: {e}')

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start',   start))
    app.add_handler(CommandHandler('help',    help_cmd))
    app.add_handler(CommandHandler('crowd',   crowd))
    app.add_handler(CommandHandler('now',     now))
    app.add_handler(CommandHandler('whoami',  whoami))
    app.add_handler(CommandHandler('status',  status))
    app.add_handler(CommandHandler('stats',   stats))
    app.add_handler(CommandHandler('history', history))
    app.add_handler(CommandHandler('photo',   photo))
    app.add_handler(MessageHandler(tg_filters.PHOTO, photo))
    print('Bot running. Ctrl + C to kill. hehehehehe')
    init_db()
    start_receiver()
    app.run_polling()

if __name__ == '__main__':
    main()