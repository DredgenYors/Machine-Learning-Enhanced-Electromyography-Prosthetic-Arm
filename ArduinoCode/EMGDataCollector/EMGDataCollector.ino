#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo setup
#define NUM_SERVOS 5
#define SERVO_MIN 150   // open
#define SERVO_MAX 600   // closed
#define SERVO_DELAY 50  // ms between loops

// Updated EMG thresholds (for ~0–400 range)
#define THRESH_LOW 120
#define THRESH_HIGH 300
const int EMG_PIN = A0;

// Gesture tracking: 0 = paper, 1 = scissors, 2 = rock
int currentGesture = 0;

// Finger mapping
int servoPins[NUM_SERVOS] = {0, 1, 2, 3, 4}; // pinky, ring, middle, index, thumb

// Data collection variables
bool collectingData = false;
unsigned long collectionStartTime = 0;
int samplesCollected = 0;
#define SAMPLES_PER_GESTURE 100

void setup() {
  Serial.begin(115200);
  Serial.println("Rock-Paper-Scissors Prosthetic (3-Gesture Mode)");

  pwm.begin();
  pwm.setPWMFreq(60);
  delay(10);

  setPaper();
}

void loop() {
  // Handle serial commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    processCommand(command);
  }

  // Data collection loop
  if (collectingData) {
    int rawValue = analogRead(EMG_PIN);
    float voltage = rawValue * (5.0 / 1023.0);
    unsigned long timestamp = millis();

    // Stream CSV
    Serial.print("DATA,");
        Serial.print(currentGesture);
        Serial.print(",");
        Serial.print(rawValue);
        Serial.print(",");
        Serial.println(voltage, 3);

    // Move servos based on EMG thresholds
    if (rawValue > THRESH_HIGH) {
      setRock();
    } else if (rawValue > THRESH_LOW) {
      setScissors();
    } else {
      setPaper();
    }

    samplesCollected++;

    if (samplesCollected >= SAMPLES_PER_GESTURE) {
      collectingData = false;
      Serial.println("COLLECTION_COMPLETE");
      moveToGesture("paper");
    }

    delay(50); // 20Hz
  }
}

void processCommand(String command) {
  if (command.startsWith("gesture=")) {
    String g = command.substring(8);
    if (g == "paper") currentGesture = 0;
    else if (g == "scissors") currentGesture = 1;
    else if (g == "rock") currentGesture = 2;

    Serial.print("Gesture set to: ");
    Serial.println(currentGesture);
  }

  else if (command == "start") {
    collectingData = true;
    samplesCollected = 0;
    collectionStartTime = millis();
    Serial.println("Starting data collection...");
  }

  else if (command == "stop") {
    collectingData = false;
    Serial.println("Stopping data collection...");
  }
}

// ===================== Gesture Functions ======================
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

void moveToGesture(String gesture) {
  if (gesture == "rock") {
    setRock();
  } else if (gesture == "paper") {
    setPaper();
  } else if (gesture == "scissors") {
    setScissors();
  } else {
    setPaper();
  }
  delay(500);
}