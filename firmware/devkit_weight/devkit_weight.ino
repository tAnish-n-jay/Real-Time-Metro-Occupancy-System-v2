#include <WiFi.h>
#include <HTTPClient.h>
#include "HX711.h"


const char* WIFI_SSID = "Tanish";
const char* WIFI_PASS = "123456789";     
const char* SERVER_URL = "http://10.215.133.24:5000";
const int DT_PIN = 16;
const int SCK_PIN = 4;
const float CALIBRATION_FACTOR = 556.8172;


const float DEADBAND_G = 5.0;        
const unsigned long SEND_EVERY_MS = 2000;

HX711 scale;
unsigned long lastSend = 0;

void connectWiFi() {
  Serial.print("Connecting to WiFi ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("\nConnected. ESP32 IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi failed, will retry.");
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  scale.begin(DT_PIN, SCK_PIN);
  scale.set_scale(CALIBRATION_FACTOR);
  Serial.println("Platform must be EMPTY. Taring in 3 seconds...");
  delay(3000);
  scale.tare(20);
  Serial.println("Tare done.");
  connectWiFi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
    return;
  }

  if (millis() - lastSend >= SEND_EVERY_MS) {
    lastSend = millis();

    float grams = scale.get_units(10);   
    if (grams < DEADBAND_G) grams = 0;  

    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(3000);
    String body = "{\"grams\": " + String(grams, 1) + "}";
    int code = http.POST(body);
    Serial.print("grams=");
    Serial.print(grams, 1);
    Serial.print("  POST status=");
    Serial.println(code);  
    http.end();
  }
}