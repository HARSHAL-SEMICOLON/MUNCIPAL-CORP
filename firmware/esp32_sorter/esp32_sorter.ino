/*
 * esp32_sorter -- device side of the sorting protocol.
 *
 * Implements the same line protocol as hardware/protocol.py, which is the
 * contract between this sketch and the Python system. If you change a
 * message here, change it there; the tests in tests/test_hardware.py hold
 * the host side to it.
 *
 *     host -> board    #<seq> <VERB> [args]
 *     board -> host    #<seq> OK [detail]
 *                      #<seq> ERR <code> [detail]
 *                      !EVT <NAME> [detail]
 *
 * Three rules this sketch follows, and each matters more than it looks:
 *
 * 1. ALWAYS echo the sequence number you were given. The host discards any
 *    reply whose sequence does not match what it is waiting for. Reply with
 *    the wrong number and your ACK is thrown away; reply with none and every
 *    command times out.
 *
 * 2. Only answer OK once the servo has actually ARRIVED. Answering when the
 *    command is accepted rather than completed is the classic mistake -- the
 *    host then believes an item was diverted while the arm is still moving,
 *    and the next item goes into the wrong bin.
 *
 * 3. The emergency stop cuts motor power in HARDWARE. The check below is a
 *    second line of defence, not the first. Never rely on firmware alone to
 *    stop a machine that can hurt someone.
 *
 * Board: ESP32 dev module. Libraries: ESP32Servo.
 */

#include <ESP32Servo.h>

// ---------------------------------------------------------------------------
//  PINS -- yours to change. These are placeholders, not a wiring diagram.
// ---------------------------------------------------------------------------
const int PIN_SERVO      = 13;   // diverter servo signal
const int PIN_MOTOR_PWM  = 25;   // conveyor motor driver enable (PWM)
const int PIN_MOTOR_DIR  = 26;   // conveyor motor driver direction
const int PIN_ESTOP      = 27;   // emergency stop, active LOW, INPUT_PULLUP
const int PIN_ECHO       = 34;   // HC-SR04 echo   (bin level, optional)
const int PIN_TRIG       = 33;   // HC-SR04 trigger

// ---------------------------------------------------------------------------
//  MECHANISM
// ---------------------------------------------------------------------------
const unsigned long SERVO_SETTLE_MS = 350;   // measure this on your rig
const int  MOTOR_SPEED   = 180;              // 0-255 PWM duty
const unsigned long BIN_REPORT_MS = 5000;    // how often to report bin level

Servo diverter;
bool conveyorRunning = false;
bool estopEngaged    = false;
int  lastAngle       = -1;
unsigned long lastBinReport = 0;

String inputLine;

// ---------------------------------------------------------------------------
//  Bin -> servo angle. These MUST match hardware.bin_angles in config.yaml.
//  Two sources of truth is a known hazard here; if they drift, items go to
//  the wrong bin and nothing reports an error. Change them together.
// ---------------------------------------------------------------------------
struct BinAngle { const char* name; int angle; };
const BinAngle BIN_ANGLES[] = {
  {"RECYCLING",     30},
  {"ORGANIC",       60},
  {"E_WASTE",       90},
  {"HAZARDOUS",    120},
  {"REJECT",       150},
  {"MANUAL_CHECK", 180},
};
const int BIN_COUNT = sizeof(BIN_ANGLES) / sizeof(BIN_ANGLES[0]);

int angleFor(const String& name) {
  for (int i = 0; i < BIN_COUNT; i++) {
    if (name.equals(BIN_ANGLES[i].name)) return BIN_ANGLES[i].angle;
  }
  return -1;
}

// ---------------------------------------------------------------------------
//  Replies
// ---------------------------------------------------------------------------
void replyOk(long seq, const String& detail) {
  Serial.print('#'); Serial.print(seq); Serial.print(" OK");
  if (detail.length()) { Serial.print(' '); Serial.print(detail); }
  Serial.print('\n');
}

void replyErr(long seq, const char* code, const String& detail) {
  Serial.print('#'); Serial.print(seq); Serial.print(" ERR "); Serial.print(code);
  if (detail.length()) { Serial.print(' '); Serial.print(detail); }
  Serial.print('\n');
}

void sendEvent(const char* name, const String& detail) {
  Serial.print("!EVT "); Serial.print(name);
  if (detail.length()) { Serial.print(' '); Serial.print(detail); }
  Serial.print('\n');
}

// ---------------------------------------------------------------------------
//  Actuators
// ---------------------------------------------------------------------------
void motorOn()  { digitalWrite(PIN_MOTOR_DIR, HIGH); analogWrite(PIN_MOTOR_PWM, MOTOR_SPEED); conveyorRunning = true;  }
void motorOff() { analogWrite(PIN_MOTOR_PWM, 0);                                              conveyorRunning = false; }

// Returns true once the arm has arrived. Blocking is acceptable here: this
// board does one thing, and the host is waiting for exactly this answer.
bool moveDiverter(int angle) {
  diverter.write(angle);
  delay(SERVO_SETTLE_MS);
  lastAngle = angle;
  // With a feedback servo or an endstop, verify the position HERE and return
  // false if it did not arrive -- that is what makes ERR JAM meaningful
  // rather than decorative. Without feedback this is optimistic, and the
  // host's retry-then-hold logic is the only safety net.
  return true;
}

// ---------------------------------------------------------------------------
//  Command handling
// ---------------------------------------------------------------------------
void handleCommand(const String& line) {
  if (!line.startsWith("#")) return;                 // not for us

  int firstSpace = line.indexOf(' ');
  if (firstSpace < 0) return;

  long seq = line.substring(1, firstSpace).toInt();
  String rest = line.substring(firstSpace + 1);
  rest.trim();

  int argSpace = rest.indexOf(' ');
  String verb = (argSpace < 0) ? rest : rest.substring(0, argSpace);
  String arg  = (argSpace < 0) ? ""   : rest.substring(argSpace + 1);
  arg.trim();

  // PING, STATUS and STOP stay answerable during an emergency stop: the host
  // needs to be able to ask what is going on, and STOP is never unsafe.
  if (estopEngaged && verb != "PING" && verb != "STATUS" && verb != "STOP") {
    replyErr(seq, "ESTOP", "emergency stop engaged");
    return;
  }

  if (verb == "PING") {
    replyOk(seq, "pong");

  } else if (verb == "STATUS") {
    String s = "running=" + String(conveyorRunning ? 1 : 0)
             + " estop="  + String(estopEngaged ? 1 : 0)
             + " angle="  + String(lastAngle);
    replyOk(seq, s);

  } else if (verb == "START") {
    motorOn();
    replyOk(seq, "motor on");

  } else if (verb == "STOP") {
    motorOff();
    replyOk(seq, "motor off");

  } else if (verb == "HOLD") {
    if (!arg.length()) { replyErr(seq, "BAD_COMMAND", "HOLD needs an item id"); return; }
    motorOff();
    replyOk(seq, "holding " + arg);

  } else if (verb == "SORT") {
    if (!arg.length()) { replyErr(seq, "BAD_COMMAND", "SORT needs a bin"); return; }
    int angle = angleFor(arg);
    if (angle < 0) {
      replyErr(seq, "UNKNOWN_BIN", "no angle configured for " + arg);
      return;
    }
    if (!moveDiverter(angle)) {
      replyErr(seq, "JAM", "diverter did not reach " + String(angle) + "deg");
      return;
    }
    replyOk(seq, "angle=" + String(angle));

  } else {
    replyErr(seq, "BAD_COMMAND", "unknown verb " + verb);
  }
}

// ---------------------------------------------------------------------------
//  Sensors
// ---------------------------------------------------------------------------
void checkEstop() {
  bool pressed = (digitalRead(PIN_ESTOP) == LOW);    // active LOW
  if (pressed && !estopEngaged) {
    estopEngaged = true;
    motorOff();
    sendEvent("ESTOP_ENGAGED", "button");
  } else if (!pressed && estopEngaged) {
    estopEngaged = false;
    // The belt is deliberately NOT restarted here. Clearing an emergency stop
    // must never start a machine on its own -- an operator restarts it.
    sendEvent("ESTOP_CLEARED", "button");
  }
}

long readDistanceCm() {
  digitalWrite(PIN_TRIG, LOW);  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);
  long duration = pulseIn(PIN_ECHO, HIGH, 30000);    // 30 ms ceiling
  if (duration == 0) return -1;                      // nothing in range
  return duration / 58;
}

void reportBinLevel() {
  if (millis() - lastBinReport < BIN_REPORT_MS) return;
  lastBinReport = millis();

  const long EMPTY_CM = 40;                          // sensor to empty floor
  long distance = readDistanceCm();
  if (distance < 0) return;

  long percent = 100 - (distance * 100 / EMPTY_CM);
  percent = constrain(percent, 0, 100);
  sendEvent("BIN_LEVEL", "RECYCLING " + String(percent));
}

// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);

  pinMode(PIN_MOTOR_PWM, OUTPUT);
  pinMode(PIN_MOTOR_DIR, OUTPUT);
  pinMode(PIN_ESTOP, INPUT_PULLUP);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);

  diverter.attach(PIN_SERVO);
  diverter.write(BIN_ANGLES[0].angle);
  lastAngle = BIN_ANGLES[0].angle;
  motorOff();

  delay(200);
  sendEvent("BOOT", "esp32_sorter ready");
}

void loop() {
  checkEstop();
  reportBinLevel();

  while (Serial.available()) {
    char c = (char) Serial.read();
    if (c == '\n') {
      inputLine.trim();
      if (inputLine.length()) handleCommand(inputLine);
      inputLine = "";
    } else if (c != '\r' && inputLine.length() < 120) {
      inputLine += c;
    }
  }
}
