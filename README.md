# Real-Time Metro Coach Occupancy Monitoring System

> **Multi-modal IoT system** combining weight sensing and computer vision to estimate live passenger occupancy in metro coaches — with a Telegram bot interface, analytics, and sensor mismatch detection.

---

## Overview

PhishPrint... wait, wrong project. This system monitors metro coach crowding in real time using a **dual-sensor fusion approach**:

-  **Weight-based counting** via load cell + HX711 on an ESP32 DevKit V1
-  **Vision-based counting** via ESP32-CAM + YOLOv8n person detection
- **Sensor fusion** (50-50 weighted average) for higher-confidence headcount
-  **Mismatch detection** that flags luggage-vs-people scenarios
-  **Telegram bot** for live queries, alerts, and trend analytics

Built as part of the IoT Semester End Examination at **RV University, Bengaluru**.

---

##  System Architecture

```
[Load Cell] → [HX711] → [ESP32 DevKit V1]
                                    ↓
                            HTTP POST /data (JSON)
                                    ↓
[OV2640 Camera] → [ESP32-CAM] → HTTP POST /image (JPEG)
                                    ↓
                          [Python Flask Backend]
                                    ↓
                    ┌───────────────────────────┐
                    │       Fusion Engine        │
                    │  weight_count  (50%)       │
                    │+ vision_count  (50%)       │
                    │= fused_count               │
                    │+ mismatch flag if |diff|≥2 │
                    └───────────────────────────┘
                                    ↓
                            [SQLite Database]
                             ↙           ↘
                 [Telegram Bot]     [Analytics Engine]
                (live queries)    (trends + projection)
```

---

##  Features

| Feature | Description |
|---|---|
| Dual sensing | Load cells + ESP32-CAM vision counting |
| Sensor fusion | 50-50 weighted average of both headcounts |
| Mismatch detection | Flags when sensors disagree by ≥2 people (luggage scenario) |
| Per-station logging | Occupancy logged per station as train progresses |
| Telegram bot | Live status, peak hours, next-station projection, push alerts |
| Analytics | Moving average trend projection, peak hour analysis |
| Resilience | ESP32 retry logic (3 attempts + backoff), graceful single-sensor fallback |

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Firmware (weight) | Arduino C++ — ESP32 DevKit V1 |
| Firmware (camera) | Arduino C++ — ESP32-CAM, `esp_camera.h` |
| Backend | Python + Flask |
| Image processing | OpenCV (`cv2`) |
| Person detection | YOLOv8n (ultralytics, pretrained on COCO) |
| Database | SQLite (prototype) → MySQL (production) |
| Bot | python-telegram-bot |
| Analytics | pandas + matplotlib |

---

##  Hardware

### Components

| Component | Qty | Purpose |
|---|---|---|
| ESP32 DevKit V1 | 1 | Load cell brain + WiFi + OLED + buzzer |
| 5kg Load Cell | 4 | Corner platform weight sensing |
| HX711 Amplifier | 1 | ADC for load cells |
| SSD1306 OLED | 1 | Live display |
| Active Buzzer | 1 | Overcapacity alert |
| ESP32-CAM (AI-Thinker, OV2640) | 1 | Overhead image capture |
| FTDI Programmer (USB-to-TTL, 5V) | 1 | Flashing ESP32-CAM |
| 5V 2A external power supply | 1 | Dedicated power for ESP32-CAM |

>  **Critical:** Power the ESP32-CAM from its own dedicated 5V 2A supply. Camera + WiFi draw simultaneous current spikes that cause brownouts on shared USB power.

### Platform Setup

```
         [ESP32-CAM on overhead mount]
                    ↓ looking down
         ┌─────────────────────────┐
         │     Plywood Platform    │
         │  [LC] [LC] [LC] [LC]   │  ← 4 load cells at corners
         │                         │
         │   [figures / objects]   │  ← demo "passengers"
         └─────────────────────────┘
         [ESP32 DevKit V1 + OLED below/beside]
```

---

##  Database Schema

```sql
CREATE TABLE occupancy_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    coach_id         TEXT,
    station_name     TEXT,
    weight_kg        REAL,
    weight_count     INTEGER,
    vision_count     INTEGER,
    fused_count      INTEGER,
    occupancy_pct    REAL,
    status           TEXT,             -- GOOD / MODERATE / CROWDED / FULL
    mismatch_flag    TEXT DEFAULT NULL,
    timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE stations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    station_name  TEXT,
    station_order INTEGER
);
```

---

## Telegram Bot Commands

| Command | Description |
|---|---|
| `/status` | Live occupancy — weight, vision, fused counts + mismatch flag |
| `/coach1` | Specific coach live status |
| `/station <name>` | Occupancy at a specific station |
| `/peak <station>` | Peak hours projection for that station |
| `/predict <station>` | Projected occupancy at next station (moving average) |
| `/history` | Last 24-hour trend summary |
| `/alert on` | Subscribe to FULL + mismatch push alerts |

### Sample `/status` output

```
Coach 1 @ MG Road
──────────────────
Weight estimate:  4 people
Vision estimate:  3 people
Fused estimate:   4 people
Occupancy:        67% — MODERATE
 SENSOR MISMATCH — possible luggage or occlusion
```

---

##  Occupancy Status Levels

| Occupancy | Status |
|---|---|
| 0 – 49% |  GOOD |
| 50 – 79% | MODERATE |
| 80 – 99% | CROWDED |
| 100% | FULL + Buzzer |

---

## Setup & Running

### Prerequisites

```bash
pip install flask ultralytics opencv-python python-telegram-bot pandas matplotlib
```

### 1. Start the Flask backend

```bash
python app.py
```

### 2. Flash ESP32 DevKit V1

Open `firmware/esp32_weight/esp32_weight.ino` in Arduino IDE, set your WiFi credentials and server IP, and flash.

### 3. Flash ESP32-CAM

Connect via FTDI programmer, open `firmware/esp32_cam/esp32_cam.ino`, set WiFi + server IP, flash at QVGA resolution.

### 4. Start the Telegram bot

```bash
python bot.py
```

---

## Project Structure

```
.
├── firmware/
│   ├── esp32_weight/       # DevKit V1 — load cell + OLED + retry logic
│   └── esp32_cam/          # ESP32-CAM — JPEG capture + HTTP POST
├── backend/
│   ├── app.py              # Flask server — /data + /image routes + fusion
│   ├── db.py               # SQLite helpers
│   └── fusion.py           # Fusion engine + mismatch detection
├── bot/
│   └── bot.py              # Telegram bot — commands + push alerts
├── analytics/
│   └── analytics.py        # Peak hour + moving average projection
├── schema.sql              # SQLite schema
└── README.md
```

---

## Research Paper Novelty Claims

1. **4-cell summed load platform** eliminates orientation-based calibration error
2. **Multi-modal occupancy estimation** via fusion of weight-based and vision-based headcount
3. **Sensor mismatch detection** flags luggage-vs-people scenarios that fool single-sensor systems
4. **Per-station occupancy logging** enables historical trend analysis
5. **Telegram bot interface** delivers data without a dedicated mobile app
6. **System resilience testing** — WiFi drop and server restart recovery
7. **Total cost under ₹3,500** — significant cost advantage over existing solutions

---

##  Team

**Tanish N Jay · Nithin J · Tushar R**  
AI/ML Major, BTech CSE — RV University, Bengaluru

---

##  License

MIT License — feel free to use, adapt, and build on this.