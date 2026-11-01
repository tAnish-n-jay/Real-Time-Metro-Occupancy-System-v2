import os
import pickle
import re
import difflib
from datetime import datetime, timedelta

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

with open('models/crowd_model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

TABLE = MODEL['table']
STATIONS = MODEL['stations']

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
            "Predicted from historical data, relative to this station's busy level")

HELP = ("Commands:\n"
        "/crowd <station> <day> <time> e.g. /crowd majestic mon 9am\n"
        "/now <station> typical crowd for this day and hour\n"
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
        await update.message.reply_text(f"Try again! '{args[-1]}'. Try 9, 9am, 9pm or 21:00.'")
        return 
    if day is None:
        await update.message.reply_text(f'Try again! "{args[-2]}". Try mon..sun, weekday, today or tomorrow.')
        return
    await update.message.reply_text(answer(" ".join(args[:-2]), day[0], day[1], hour))

async def now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text('Usage: /now <station>\nExample: /now majestic')
        return 
    d = datetime.now()
    await update.message.reply_text(answer(" ".join(context.args), day_type_of(d), d.strftime('%A'), d.hour))

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_cmd))
    app.add_handler(CommandHandler('crowd', crowd))
    app.add_handler(CommandHandler('now', now))
    print('Bot running. Ctrl + C to kill. hehehehehe')
    app.run_polling()

if __name__ == '__main__':
    main()
                                    