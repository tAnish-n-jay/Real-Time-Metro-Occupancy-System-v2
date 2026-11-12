#define BLYNK_TEMPLATE_ID   "TMPL319CcbeOP"       
#define BLYNK_TEMPLATE_NAME "Coach Occupancy"
#define BLYNK_AUTH_TOKEN    "BYLNK_AUTH_TOKEN"         

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <HX711.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <BlynkSimpleEsp32.h>

const char* WIFI_SSID       = "Tanish";
const char* WIFI_PASS   = "PASSWORD";
const char* SERVER_URL  = "http://10.232.x.x:5000";
const char* BLYNK_SERVER = "sgp1.blynk.cloud";           

const int   DT_PIN            = 16;
const int   SCK_PIN           = 4;
const float CALIBRATION_FACTOR = 523.8466;
const float DEADBAND_G         = 5.0;
const float FULL_G             = 1500.0;

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET   -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

const int BUZZER_PIN = 23;

const unsigned long SEND_EVERY_MS   = 2000;
const unsigned long BLYNK_EVERY_MS  = 2000;
unsigned long lastSend  = 0;
unsigned long lastBlynk = 0;
bool buzzerOn = false;

HX711 scale;

String levelLabel(float pct) {
  if (pct < 0.25) return "Low";
  if (pct < 0.50) return "Moderate";
  if (pct < 0.75) return "Crowded";
  return "Very Crowded";
}

void updateOLED(float grams, float pct) {
  int occ = (int)(pct * 100);
  display.clearDisplay();
  display.setTextColor(WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("Pattanagere");
  display.setCursor(90, 0);
  display.print(occ >= 80 ? "FULL!" : "OK");

  display.setCursor(0, 14);
  display.print("Weight: ");
  display.print(grams, 0);
  display.print(" g");

  display.setCursor(0, 26);
  display.print("Load: ");
  display.print(occ);
  display.print("% ");
  display.print(levelLabel(pct));

  display.drawRect(0, 50, 128, 10, WHITE);
  int barW = (occ * 124) / 100;
  if (barW > 124) barW = 124;
  display.fillRect(2, 52, barW, 6, WHITE);

  display.display();
}

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Wire.begin(21, 22);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED not found! Check wiring.");
    while (true);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(10, 20);
  display.println("Starting up...");
  display.display();

  scale.begin(DT_PIN, SCK_PIN);
  scale.set_scale(CALIBRATION_FACTOR);
  Serial.println("Platform EMPTY. Taring in 3 s...");
  display.clearDisplay();
  display.setCursor(10, 20);
  display.println("Taring... keep empty");
  display.display();
  delay(3000);
  scale.tare(20);
  Serial.println("Tare done.");

  display.clearDisplay();
  display.setCursor(10, 20);
  display.println("Connecting WiFi...");
  display.display();
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi failed. Running without cloud.");
    display.clearDisplay();
    display.setCursor(10, 20);
    display.println("WiFi FAILED");
    display.println("No Blynk / No Bot");
    display.display();
    delay(2000);
  } else {
    Serial.print("\nConnected. IP: ");
    Serial.println(WiFi.localIP());
    display.clearDisplay();
    display.setCursor(10, 20);
    display.println("WiFi Connected!");
    display.display();
    delay(500);

    display.clearDisplay();
    display.setCursor(10, 20);
    display.println("Connecting Blynk...");
    display.display();
    Blynk.config(BLYNK_AUTH_TOKEN, BLYNK_SERVER, 80);
    Blynk.connect(5000);  
    if (Blynk.connected()) {
      Serial.println("Blynk connected.");
    } else {
      Serial.println("Blynk failed. Bot posting still works.");
    }
  }

  display.clearDisplay();
  display.setCursor(10, 20);
  display.println("System Ready!");
  display.display();
  delay(1000);
  Serial.println("Ready.");
  Serial.print("Bot URL: ");
  Serial.println(SERVER_URL);
}

void loop() {
  Blynk.run();

  unsigned long now = millis();
  if (now - lastSend < SEND_EVERY_MS) return;
  lastSend = now;

  if (!scale.is_ready()) return;

  float grams = scale.get_units(10);
  if (grams < DEADBAND_G) grams = 0;
  float pct = grams / FULL_G;
  if (pct > 1.0) pct = 1.0;
  int occ = (int)(pct * 100);

  updateOLED(grams, pct);

  Serial.print("grams="); Serial.print(grams, 1);
  Serial.print("  occ="); Serial.print(occ);
  Serial.print("%  level="); Serial.println(levelLabel(pct));

  static bool buzzedThisCycle = false;
  if (pct >= 0.90 && !buzzedThisCycle) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(1000);
    digitalWrite(BUZZER_PIN, LOW);
    buzzedThisCycle = true;
  } else if (pct < 0.80) {
    buzzedThisCycle = false;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(3000);
  String body = "{\"grams\": " + String(grams, 1) + "}";
  int code = http.POST(body);
  Serial.print("  POST status="); Serial.println(code);
  if (code == -1) {
  Serial.print("Cannot reach bot. Expected IP: ");
  Serial.println(SERVER_URL);
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
  }
  http.end();

  if (now - lastBlynk >= BLYNK_EVERY_MS && Blynk.connected()) {
    lastBlynk = now;
    Blynk.virtualWrite(V0, grams);  // kg
    Blynk.virtualWrite(V1, grams / 65.0);  //OCC        
    Blynk.virtualWrite(V2, levelLabel(pct));  // string
  }
}