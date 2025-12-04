#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// ====== Servo + EMG Setup ======
#define NUM_SERVOS 5
#define SERVO_MIN 150
#define SERVO_MAX 600
int servoPins[NUM_SERVOS] = {0, 1, 2, 3, 4};  // pinky, ring, middle, index, thumb

const int EMG_PIN = A0;

// Gesture: 0 = paper, 1 = scissors, 2 = rock, -1 = neutral
int currentGesture = -1;
unsigned long lastGestureChange = 0;
const unsigned long HOLD_MS = 3000;   // hold each predicted gesture for 3s

void setup() {
  Serial.begin(115200);
  pwm.begin();
  pwm.setPWMFreq(60);
  delay(10);

  setPaper();  // neutral pose
  Serial.println("[Arduino] Realtime EMG + ML gesture listener READY");
}

void loop() {
  // -------------------------------------------------
  // 1) STREAM EMG SAMPLE TO PYTHON
  // -------------------------------------------------
  int rawValue = analogRead(EMG_PIN);
  float voltage = rawValue * (5.0 / 1023.0);

  // Format expected by Python: DATA,<raw>,<voltage>
  Serial.print("DATA,");
  Serial.print(rawValue);
  Serial.print(",");
  Serial.println(voltage, 3);

  // -------------------------------------------------
  // 2) LISTEN FOR GESTURE PREDICTION FROM PYTHON
  // -------------------------------------------------
  if (Serial.available() > 0) {
    int g = Serial.parseInt();   // grabs an integer like 0,1,2 from the stream

    if (g == 0 || g == 1 || g == 2) {
      if (g != currentGesture) {
        currentGesture = g;
        lastGestureChange = millis();

        Serial.print("[Arduino] Received ML gesture = ");
        Serial.println(currentGesture);

        applyGesture(currentGesture);
      }
    } else {
      // uncomment for debugging “junk” input:
      // Serial.print("[Arduino] Ignoring value: ");
      // Serial.println(g);
    }
  }

  // -------------------------------------------------
  // 3) OPTIONAL AUTO-RESET AFTER HOLD_MS
  // -------------------------------------------------
  if (currentGesture != -1 && (millis() - lastGestureChange) > HOLD_MS) {
    Serial.println("[Arduino] HOLD over, resetting to PAPER");
    setPaper();
    currentGesture = -1;
  }

  delay(20);   // ~50 Hz streaming
}

// ==================== GESTURE HELPERS ====================
void applyGesture(int g) {
  if (g == 0) {
    Serial.println("[Arduino] -> PAPER");
    setPaper();
  } else if (g == 1) {
    Serial.println("[Arduino] -> SCISSORS");
    setScissors();
  } else if (g == 2) {
    Serial.println("[Arduino] -> ROCK");
    setRock();
  } else {
    setPaper();
  }
}

void setPaper() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    pwm.setPWM(servoPins[i], 0, SERVO_MIN);
  }
}

void setScissors() {
  pwm.setPWM(servoPins[0], 0, SERVO_MAX);
  pwm.setPWM(servoPins[1], 0, SERVO_MAX);
  pwm.setPWM(servoPins[2], 0, SERVO_MIN);
  pwm.setPWM(servoPins[3], 0, SERVO_MIN);
  pwm.setPWM(servoPins[4], 0, SERVO_MAX);
}

void setRock() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    pwm.setPWM(servoPins[i], 0, SERVO_MAX);
  }
}
