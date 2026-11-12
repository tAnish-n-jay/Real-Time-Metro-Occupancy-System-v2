#include "HX711.h"

const int DT_PIN  = 16;   // change to your DT pin
const int SCK_PIN = 4;    // change to your SCK pin

HX711 scale;
float factor = 1.0;

void setup() {
  Serial.begin(115200);
  scale.begin(DT_PIN, SCK_PIN);
  Serial.println("Platform must be EMPTY. Taring in 3 seconds...");
  delay(3000);
  scale.set_scale();       // factor 1 for now
  scale.tare(20);          // average 20 readings as zero
  Serial.println("Place a KNOWN weight on the platform, then type its weight in grams (e.g. 500) and press Enter.");
}

void loop() {
  if (Serial.available()) {
    float known = Serial.parseFloat();
    while (Serial.available()) Serial.read();
    if (known > 0) {
      long raw = scale.get_value(20);        // average reading minus tare
      factor = raw / known;
      Serial.print("raw="); Serial.print(raw);
      Serial.print("  CALIBRATION_FACTOR="); Serial.println(factor, 4);
      scale.set_scale(factor);
    }
  }
  if (factor != 1.0) {
    Serial.print("grams: "); Serial.println(scale.get_units(10), 1);
    delay(300);
  }
}