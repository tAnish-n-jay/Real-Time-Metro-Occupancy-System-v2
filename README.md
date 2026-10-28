# Real-Time Metro Occupancy System v2

A Telegram bot that tells you how crowded a Namma Metro (Bengaluru) coach is likely to be at a given station, day and time. An ML model trained on real hourly ridership data does the predicting.

## What it does

- `/crowd <station> <day> <time>` replies with the predicted crowd level for that station and time
- `/now <station>` replies with the predicted crowd level right now
- `/start` and `/help` explain usage

## How it works

```
BMRCL hourly ridership (RTI data)
        |
   clean + build crowd index        notebooks/01_model.ipynb
        |
   train + evaluate model           (held-out days, not random rows)
        |
   models/crowd_model.joblib
        |
   Telegram bot loads the model     bot/bot.py
```

**Crowd index** = ridership at a station in an hour / that station's busiest hour. It is a *relative* crowd level (0 to 100%), not a count of people on a train. Ridership data counts passengers boarding, so it cannot give true coach occupancy on its own.

## Dataset

Hourly station-wise ridership for Namma Metro, obtained by RTI and cleaned into CSV/Parquet by Vivek Mathew:
https://github.com/Vonter/bmrcl-ridership-hourly

Clone it into `data/bmrcl/` (the folder is not committed to this repo).

## Setup (Windows)

```
python -m venv venv
venv\Scripts\activate
pip install pandas numpy scikit-learn joblib pyarrow jupyter ipykernel python-telegram-bot python-dotenv
python -m ipykernel install --user --name metro --display-name "metro venv"

git clone https://github.com/Vonter/bmrcl-ridership-hourly data\bmrcl
```

Create a `.env` file in the project root (never commit it):

```
TELEGRAM_TOKEN=your_token_from_botfather
```

## Project structure

```
Real-Time-Metro-Occupancy-System-v2/
├── README.md
├── PLAN.md                  # phases and checklist
├── .env                     # TELEGRAM_TOKEN (gitignored)
├── data/
│   └── bmrcl/               # cloned dataset (gitignored)
├── notebooks/
│   └── 01_model.ipynb       # load, clean, train, evaluate, save model
├── models/
│   └── crowd_model.joblib   # saved model + station list
├── bot/
│   └── bot.py               # Telegram bot (runs as a normal script)
├── src/
│   └── inspect_data.py      # quick look at the dataset files
├── tests/
│   └── vision_test.py       # archived vision experiment (see below)
├── test_images/             # archived vision experiment images
└── docs/                    # results, screenshots, paper notes
```

## Archived: vision experiment

The original plan used an ESP32-CAM and YOLOv8n to count people. An informal test on a handful of stock photos (not our own camera data) showed:

- front-on photos: counts matched for 2, 3 and 4 people
- high-angle photo: 7 of 10 people detected (misses were in a tight huddle)
- straight-down photo: 2 of 7 detected

This suggests a pretrained COCO detector struggles from straight overhead. The vision feature was dropped for this version because of the time limit, before it was tested on the real camera and demo objects. It is not evidence about the final hardware setup.

## Limitations

- The dataset covers only about 19 days, so it cannot capture monthly or seasonal patterns
- The crowd index is relative to each station's own peak, so values are not comparable as passenger counts between stations
- Predictions are expected patterns, not live measurements
- Model quality is estimated on held-out days, which is a small test set

## Future Implementations

- Estimate passengers on board between two stations from the station-pair flows, and divide by train capacity for a more realistic occupancy percentage
- Feed a live ESP32 load-cell reading into the bot as a "live" value next to the prediction
- `/peak <station>` command showing the busiest hours
