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
from fastapi import FastAPI, Request
from datetime import datetime, timedelta
import sqlite3
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters as tg_filters
from functools import wraps
import numpy as np
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackQueryHandler

app = FastAPI()
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
    'nano'  : YOLO('models/yolov8n.pt'),   
    'small' : YOLO('models/yolov8s.pt'),  
    'medium': YOLO('models/yolov8m.pt'),  
    'large' : YOLO('models/yolov8l.pt'),  
}
print('[YOLO] All models ready.')

def pick_model() -> tuple:
    """Pick YOLO model based on current weight sensor reading."""
    sensor_online = (
        LIVE['grams'] is not None and
        time.time() - LIVE['ts'] <= STALE_SECOND
    )

    if not sensor_online:
        key, path, name = 'medium', 'models/yolov8m.pt', 'YOLOv8m (sensor offline — default)'
    else:
        pct = LIVE['grams'] / FULL_G
        if pct < 0.25:
            key, path, name = 'nano',   'models/yolov8n.pt', 'YOLOv8n (Low)'
        elif pct < 0.50:
            key, path, name = 'small',  'models/yolov8s.pt', 'YOLOv8s (Moderate)'
        elif pct < 0.75:
            key, path, name = 'medium', 'models/yolov8m.pt', 'YOLOv8m (Crowded)'
        else:
            key, path, name = 'large',  'models/yolov8l.pt', 'YOLOv8l (Very Crowded)'

    if key not in _yolo_models:
        print(f'[YOLO] Loading {name}...')
        _yolo_models[key] = YOLO(path)

    return _yolo_models[key], name

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

@app.post('/weight')
async def receive_weight(request: Request):
    try:
        body = await request.body()
        if not body:
            return {'status': 'error', 'detail': 'empty body'}
        data = json.loads(body)
        LIVE['grams'] = float(data['grams'])
        LIVE['ts'] = time.time()
        LIVE['source'] = 'weight'
        log_reading(LIVE['grams'])
        return {'status': 'ok'}
    except Exception as e:
        print(f'[WEIGHT] Parse failed : {e}')
        return {'status': 'error', 'detail': str(e)}

@app.post('/image')
async def receive_image(request: Request):
    img_bytes = await request.body()
    try:
        content_type = request.headers.get('Content-Type', '')
        if 'octet-stream' in content_type:
            w = int(request.headers.get('X-Frame-Width', 320))
            h = int(request.headers.get('X-Frame-Height', 240))
            arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape((h, w, 2))
            r = (arr[:,:,1] & 0xF8)
            g = ((arr[:,:,1] & 0x07) << 5) | ((arr[:,:,0] & 0xE0) >> 3)
            b = (arr[:,:,0] & 0x1F) << 3
            img = Image.fromarray(np.stack([r, g, b], axis=2).astype(np.uint8))
        else:
            buf = _io.BytesIO(img_bytes)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')

        img_array = np.array(img)
        model, mname = pick_model()

        def run_yolo():
            return model(
                img_array,
                verbose=False,
                classes=PERSON_AND_BAGS,
                conf=0.2,
                iou=0.4
            )
        results = await asyncio.to_thread(run_yolo)

        person_count = sum(
            1 for box in results[0].boxes
            if int(box.cls) == 0
        )
        bag_count = sum(
            1 for box in results[0].boxes
            if int(box.cls) in [24, 25, 26, 28]
        )
        LIVE['grams']  = person_count * (FULL_G / MAX_PERSONS)
        LIVE['ts']     = time.time()
        LIVE['source'] = 'camera'
        log_reading(LIVE['grams'])

        print(f'[CAM] {person_count} people, {bag_count} bags')
        return {"status": "ok", "people": person_count, "bags": bag_count, "model": mname}
    except Exception as e:
        print(f'[CAM] YOLO failed: {e}')
        return {'status' : 'error', 'detail' : str(e)}
def start_receiver():
    import uvicorn
    threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={'host': '0.0.0.0', 'port': 5000, 'log_level': 'warning'},
        daemon=True
    ).start()
    print('[HTTP] FastAPI receiverr started at port 5000')

def level(c: float) -> str:
    return 'Low' if c < 0.25 else 'Moderate' if c < 0.5 else 'Crowded' if c < 0.75 else 'Very Crowded'

def live_text() -> str:
    if LIVE['grams'] is None or time.time() - LIVE['ts'] > STALE_SECOND:
        return 'Prototype coach: offline (no recent reading).'
    pct    = max(0.0, min(1.0, LIVE['grams'] / FULL_G))
    source = {'camera': 'camera', 'weight': 'weight sensor', 'combined': 'combined'}.get(
        LIVE['source'], LIVE['source'])
    return (f"Prototype coach (live): {pct:.0%} ({level(pct)}) "
            f"| {LIVE['grams']:.0f}g | via {source}")

async def build_live_text() -> str:
    now_str = datetime.now().strftime('%H:%M:%S')
    if LIVE['grams'] is None or time.time() - LIVE['ts'] > STALE_SECOND:
        return f'Pattanagere - Live\n\n Sensor offline\n {now_str}'
    pct = max(0.0, min(1.0, LIVE['grams'] / FULL_G))
    source = {'camera': 'cam', 'weight': 'weight'}.get(LIVE['source'], '')
    return (
        f"Pattanager - Live\n"
        f"Load: {pct:0%} ({level(pct)})\n"
        f"Weight: {LIVE['grams']:.0f}g\n"
        f"Source: {source} {LIVE['source']}\n\n"
        f'Updated: {now_str}'
    )
def live_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton('Refresh', callback_data='live_refresh')
    ]])

async def live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = await build_live_text()
    msg = await update.message.reply_text(text, reply_markup=live_keyboard())

    for _ in range(8):
        await asyncio.sleep(15)
        new_text = await build_live_text()
        try:
            await msg.edit_text(new_text, reply_markup=live_keyboard())
        except Exception:
            break
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
    return (f'Station: {st}\n{day_label}, {hour_str(hour)}\n'
            f'Crowd level: {c:.0%} ({level(c)}){note}\n\n'
            "Predicted from historical data. Not based on live count.")

HELP = ("Commands:\n"
        "/crowd <station> <day> <time>  e.g. /crowd majestic mon 9am\n"
        "/now <station>  live prototype reading + typical crowd\n"
        "/live  live coach occupancy with auto-refresh\n"
        "Just send a photo to count people and estimate load!\n\n"
        "Admin only:\n"
        "/status  prototype health and DB info\n"
        "/stats   load summary by day type\n"
        "/history chart of last 60 readings\n"
        "/showdb show last N  readings in chat e.g. /showdb 20\n"
        "/export download readings as CSV\n"
        "/senddb download raw SQLite file\n\n"
        "Days: mon..sun, weekday, today, tomorrow\n"
        "Time: 9, 9am, 6pm, 23:00")

QUICK_STATIONS = [
    "Pattanagere", "Majestic", "Whitefield",
    "Indiranagar", "MG Road", "Cubbon Park",
    "Vijayanagar", "Yeshwanthpur", "Attiguppe", "Electronic City"
]

def station_keyboard(callback_prefix: str) -> InlineKeyboardMarkup:
    buttons = []
    row = []
    for i, station in enumerate(QUICK_STATIONS):
        row.append(InlineKeyboardButton(
            station, callback_data=f'{callback_prefix} s{i}'
        ))
        if len(row) == 2:
            buttons.append(row)
            row = []

    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton('Cancel', callback_data='cancel')])
    return InlineKeyboardMarkup(buttons)

def day_keyboard(callback_prefix: str, sidx: str) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("Today",    callback_data=f"{callback_prefix} {sidx} today"),
         InlineKeyboardButton("Tomorrow", callback_data=f"{callback_prefix} {sidx} tomorrow")],
        [InlineKeyboardButton("Monday",   callback_data=f"{callback_prefix} {sidx} mon"),
         InlineKeyboardButton("Tuesday",  callback_data=f"{callback_prefix} {sidx} tue"),
         InlineKeyboardButton("Wednesday",callback_data=f"{callback_prefix} {sidx} wed")],
        [InlineKeyboardButton("Thursday", callback_data=f"{callback_prefix} {sidx} thu"),
         InlineKeyboardButton("Friday",   callback_data=f"{callback_prefix} {sidx} fri"),
         InlineKeyboardButton("Saturday", callback_data=f"{callback_prefix} {sidx} sat"),
         InlineKeyboardButton("Sunday",   callback_data=f"{callback_prefix} {sidx} sun")],
        [InlineKeyboardButton("Back",  callback_data=f"{callback_prefix} back"),
         InlineKeyboardButton("Cancel", callback_data="cancel")]
    ]
    return InlineKeyboardMarkup(buttons)

def time_keyboard(callback_prefix: str, sidx: str, day: str) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("6am",  callback_data=f"{callback_prefix} {sidx} {day} 6"),
         InlineKeyboardButton("7am",  callback_data=f"{callback_prefix} {sidx} {day} 7"),
         InlineKeyboardButton("8am",  callback_data=f"{callback_prefix} {sidx} {day} 8"),
         InlineKeyboardButton("9am",  callback_data=f"{callback_prefix} {sidx} {day} 9")],
        [InlineKeyboardButton("10am", callback_data=f"{callback_prefix} {sidx} {day} 10"),
         InlineKeyboardButton("11am", callback_data=f"{callback_prefix} {sidx} {day} 11"),
         InlineKeyboardButton("12pm", callback_data=f"{callback_prefix} {sidx} {day} 12"),
         InlineKeyboardButton("1pm",  callback_data=f"{callback_prefix} {sidx} {day} 13")],
        [InlineKeyboardButton("2pm",  callback_data=f"{callback_prefix} {sidx} {day} 14"),
         InlineKeyboardButton("3pm",  callback_data=f"{callback_prefix} {sidx} {day} 15"),
         InlineKeyboardButton("4pm",  callback_data=f"{callback_prefix} {sidx} {day} 16"),
         InlineKeyboardButton("5pm",  callback_data=f"{callback_prefix} {sidx} {day} 17")],
        [InlineKeyboardButton("6pm",  callback_data=f"{callback_prefix} {sidx} {day} 18"),
         InlineKeyboardButton("7pm",  callback_data=f"{callback_prefix} {sidx} {day} 19"),
         InlineKeyboardButton("8pm",  callback_data=f"{callback_prefix} {sidx} {day} 20"),
         InlineKeyboardButton("9pm",  callback_data=f"{callback_prefix} {sidx} {day} 21")],
        [InlineKeyboardButton("Back",   callback_data=f"{callback_prefix} {sidx} back"),
         InlineKeyboardButton("Cancel",  callback_data="cancel")]
    ]
    return InlineKeyboardMarkup(buttons)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Namma Metro Crowd Predictor.\n\n' + HELP)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)

async def crowd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    return await update.message.reply_text(
        "Crowd Prediction\nSelect a station:",
        reply_markup=station_keyboard("crowd")
    )

async def now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        d = datetime.now()
        query = " ".join(context.args)
        matches = find_station(query)
        parts = []
        if len(matches) == 1 and matches[0] == DEMO_STATION:
            parts.append(live_text())
        parts.append(answer(query, day_type_of(d), d.strftime('%A'), d.hour))
        if len(matches) == 1 and matches[0] != DEMO_STATION:
            parts.append(f'(Live prototype coach is only at {DEMO_STATION.split(',')[-1].strip()}.)')
        await update.message.reply_text('\n\n'.join(parts))
        return 

    await update.message.reply_text(
        f"{live_text()}\n\n Or select a station for prediction:",
        reply_markup=station_keyboard("now")
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == 'live_refresh':
        new_text = await build_live_text()
        await query.edit_message_text(new_text, reply_markup=live_keyboard())
        return 
    
    if data == 'cancel':
        await query.edit_message_text('cancelled')
        return
    parts = data.split(" ")
    prefix = parts[0]

    def get_station(sidx: str) -> str:
        try: 
            return QUICK_STATIONS[int(sidx[1:])]
        except Exception:
            return sidx
    if prefix == 'crowd':
        if len(parts) == 2 and parts[1] == 'back':
            await query.edit_message_text(
                'Crowd Prediction\n Select a station',
                reply_markup=station_keyboard('crowd')
            )
        elif len(parts) == 2:
            sidx = parts[1]
            station = get_station(sidx)
            await query.edit_message_text(
                f'Station: {station}\nSelect day:',
                reply_markup=day_keyboard('crowd',sidx)
            )

        elif len(parts) == 3 and parts[2] == 'back':
            sidx = parts[1]
            station = get_station(sidx)
            await query.edit_message_text(
                f'Station: {station}\n Select day:',
                reply_markup=day_keyboard('crowd', sidx)
            )
        elif len(parts) == 3:
            sidx, day = parts[1], parts[2]
            station = get_station(sidx)
            await query.edit_message_text(
                f'Station: {station}\n Day: {day}\n Select time:',
                reply_markup=time_keyboard('crowd',sidx, day)
            )
        elif len(parts) == 4:
            sidx, day, hour = parts[1], parts[2], int(parts[3])
            station = get_station(sidx)
            parsed = parse_day(day)
            if parsed is None:
                await query.edit_message_text('Invalid day.')
                return
            result = answer(station, parsed[0], parsed[1], hour)
            await query.edit_message_text(
                result, 
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton('Check another', callback_data='crowd back')
                ]])
            )
    elif prefix == 'now':
            if len(parts) == 2 and parts[1] == 'back':
                await query.edit_message_text(
                    f'{live_text()}\n Select a station',
                    reply_markup=station_keyboard('now')
                )
            elif len(parts) == 2:
                sidx = parts[1]
                station = get_station(sidx)
                d = datetime.now()
                parts_ = []
                matches = find_station(station)
                if matches and matches[0] == DEMO_STATION:
                    parts_.append(live_text())
                parts_.append(answer(station, day_type_of(d), d.strftime('%A'), d.hour))
                await query.edit_message_text(
                    '\n\n'.join(parts_),
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton('Check another', callback_data='now back')
                    ]])
                )
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
async def export_db(update: Update, context: ContextTypes.DEFAULT_TYPE):
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, ts, grams, percent FROM readings ORDER BY ts DESC"
    ).fetchall()
    con.close()

    if not rows:
        await update.message.reply_text("No readings in DB yet.")
        return
    
    import io
    import csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['id', 'timestamp', 'grams', 'percent'])
    writer.writerows(rows)
    buf.seek(0)

    await update.message.reply_document(
        document=buf.getvalue().encode(),
        filename=f"readings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        caption=f"{len(rows)} readings exported"
    )
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
    df['ts']       = pd.to_datetime(df['ts'], format='mixed')
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
async def show_db(update: Update, context: ContextTypes.DEFAULT_TYPE):
    limit = 10
    if context.args:
        try:
            limit = min(int(context.args[0]), 50)  
        except ValueError:
            pass

    con  = sqlite3.connect(DB_PATH)
    rows = con.execute(
        f"SELECT ts, grams, percent FROM readings ORDER BY ts DESC LIMIT {limit}"
    ).fetchall()
    con.close()

    if not rows:
        await update.message.reply_text("No readings yet.")
        return

    lines = [f"Last {len(rows)} readings:\n"]
    lines.append(f"{'Timestamp':<22} {'Grams':>7} {'Load':>6}")
    lines.append('-' * 38)
    for ts, grams, pct in rows:
        lines.append(f"{ts:<22} {grams:>7.0f} {pct:>6.0%}")

    await update.message.reply_text(
        "```\n" + "\n".join(lines) + "\n```",
        parse_mode="Markdown"
    )
@admin_only
async def send_db_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_document(
        document=open(DB_PATH, 'rb'),
        filename='live.db',
        caption='Raw SQLite database — open with DB Browser for SQLite'
    )
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
    df["ts"]  = pd.to_datetime(df["ts"], format='mixed')
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

    def run_yolo():
        model, model_name = pick_model()
        img     = Image.open(_io.BytesIO(buf)).convert('RGB')
        results = model(
            np.array(img),
            verbose=False,
            classes=PERSON_AND_BAGS,
            conf=0.1,
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
        pct = min(100, round(person_count / MAX_PERSONS * 100))
        lbl = level(pct / 100)

        if LIVE['grams'] is None or time.time() - LIVE['ts'] > STALE_SECOND:
            weight_line = 'Weight sensor: offline'
        else:
            weight_pct  = max(0.0, min(1.0, LIVE['grams'] / FULL_G))
            weight_line = f"Weight sensor: {LIVE['grams']:.0f}g ({weight_pct:.0%})"

        await update.message.reply_photo(
            photo=annotated_img,
            caption=(
                f"Passengers: {person_count} detected\n"
                f"Bags/Luggage: {bag_count} detected\n"
                f"Load based on Camera: {pct}% ({lbl})\n"
                f"{weight_line}\n"
            )
        )

    except Exception as e:
        await update.message.reply_text(f'Detection failed: {e}')

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_cmd))
    app.add_handler(CommandHandler('crowd', crowd))
    app.add_handler(CommandHandler('now', now))
    app.add_handler(CommandHandler('live', live))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(CommandHandler('whoami', whoami))
    app.add_handler(CommandHandler('status', status))
    app.add_handler(CommandHandler('stats', stats))
    app.add_handler(CommandHandler('history', history))
    app.add_handler(CommandHandler('photo', photo))
    app.add_handler(CommandHandler('export', export_db))
    app.add_handler(CommandHandler('showdb', show_db))
    app.add_handler(CommandHandler('senddb', send_db_file))
    app.add_handler(MessageHandler(tg_filters.PHOTO, photo))
    print('Bot running. Ctrl + C to kill. hehehehehe')
    init_db()
    start_receiver()
    app.run_polling()

if __name__ == '__main__':
    main()