/*
  Controlador ESP32 para Roomba serie 600 (iRobot Open Interface - OI)
  ---------------------------------------------------------------------
  - Fija la velocidad de cada rueda (mm/s) vía DRIVE DIRECT (opcode 145)
  - Calcula la odometría (x, y, theta) integrando los encoders de las ruedas
  - Lee el estado de los bumpers y DETIENE el robot si detecta contacto físico
  - Expone una pequeña API HTTP en JSON para que una app externa (la página
    web en Python) pueda leer el estado y fijar velocidades

  CONEXIONES (puerto Mini-DIN de 7 pines del Roomba):
    Pin 3: Vpwr   (NO alimentar el ESP32 directamente sin regulador adecuado)
    Pin 4: GND
    Pin 5: RXD del Roomba   <- TX del ESP32
    Pin 6: TXD del Roomba   -> RX del ESP32
    Pin 7: BRC (wake)       -> pulso a GND para "despertar" el Roomba

  IMPORTANTE - NIVELES LÓGICOS:
  El puerto OI del Roomba trabaja en TTL de 0-5V y el ESP32 en 3.3V.
  Usa un conversor de nivel lógico (o como mínimo un divisor resistivo)
  en la línea RXD->RX del ESP32 para no dañar el pin. La salida TX del
  ESP32 (3.3V) suele funcionar hacia RXD del Roomba, pero un level
  shifter bidireccional es la opción segura.

  Se usa Serial2 del ESP32 (por defecto GPIO16=RX2, GPIO17=TX2) para
  hablar con el Roomba, dejando el USB (Serial) libre para depuración.

  Librerías necesarias (Arduino IDE / PlatformIO):
    - WiFi.h y WebServer.h (incluidas en el core de ESP32)
    - ArduinoJson (instalar desde el gestor de librerías)
*/

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <esp_sleep.h>

// ---------- Configuración de red ----------
const char* WIFI_SSID = "Sarita";
const char* WIFI_PASSWORD = "bubududu";

// ---------- Configuración de pines ----------
#define ROOMBA_RX_PIN 16  // RX2 del ESP32 <- TXD del Roomba (vía divisor/level shifter)
#define ROOMBA_TX_PIN 17  // TX2 del ESP32 -> RXD del Roomba
#define WAKE_PIN 25       // GPIO usado para el pulso de "despertar" (BRC)
#define BUMPER_LED_PIN 2  // LED integrado, se enciende si el robot se para por choque

HardwareSerial RoombaSerial(2);  // UART2
WebServer server(80);

// ---------- Opcodes del Open Interface ----------
enum OICommand : uint8_t {
  OI_START = 128,
  OI_SAFE = 131,
  OI_DRIVE_DIRECT = 145,
  OI_QUERY_LIST = 149,
};

// Paquetes de sensores solicitados en cada ciclo (ver OI spec Roomba 500/600):
//   7  -> Bumps and Wheel Drops   (1 byte)
//   43 -> Left Encoder Counts     (2 bytes, con signo)
//   44 -> Right Encoder Counts    (2 bytes, con signo)
//   22 -> Voltage                 (2 bytes, mV)
//   25 -> Battery Charge          (2 bytes, mAh)
//   26 -> Battery Capacity        (2 bytes, mAh)
const uint8_t SENSOR_PACKETS[] = { 7, 43, 44, 22, 25, 26 };
const uint8_t SENSOR_PACKETS_COUNT = sizeof(SENSOR_PACKETS);
const uint8_t SENSOR_RESPONSE_BYTES = 1 + 2 + 2 + 2 + 2 + 2;  // = 11

// ---------- Parámetros físicos del Roomba serie 600 ----------
const float WHEEL_DIAMETER_MM = 72.0;
const float COUNTS_PER_REV = 508.8;  // cuentas de encoder por vuelta de rueda
const float WHEEL_BASE_MM = 235.0;   // distancia entre ruedas (recalibrar si hace falta)
const float MM_PER_COUNT = (PI * WHEEL_DIAMETER_MM) / COUNTS_PER_REV;

// ---------- Estado global ----------
struct Pose {
  float x_mm = 0;
  float y_mm = 0;
  float theta_rad = 0;
} pose;

struct Sensors {
  bool bumpLeft = false;
  bool bumpRight = false;
  bool wheelDropLeft = false;
  bool wheelDropRight = false;
  bool wheelDropCaster = false;
  float voltage_v = 0;
  uint16_t battery_charge_mah = 0;
  uint16_t battery_capacity_mah = 0;
  float battery_pct = 0;
  bool stoppedByBump = false;
} sensors;

int16_t targetLeftSpeed_mms = 0;
int16_t targetRightSpeed_mms = 0;

float wheelTrimFactor = 1.0;  // multiplica la velocidad de la rueda derecha para compensar asimetria mecanica entre ruedas

int16_t lastLeftEncoder = 0;
int16_t lastRightEncoder = 0;
bool firstEncoderReading = true;

unsigned long lastSensorPoll = 0;
const unsigned long SENSOR_POLL_INTERVAL_MS = 50;  // ~20 Hz

// ==================================================================
//  BAJO NIVEL - COMUNICACIÓN CON EL ROOMBA
// ==================================================================

void wakeRoomba() {
  pinMode(WAKE_PIN, OUTPUT);
  digitalWrite(WAKE_PIN, HIGH);
  delay(100);
  digitalWrite(WAKE_PIN, LOW);
  delay(500);
  digitalWrite(WAKE_PIN, HIGH);
  delay(100);
}

void roombaStart() {
  RoombaSerial.write(OI_START);
  delay(50);
  RoombaSerial.write(OI_SAFE);  // modo seguro: conserva las protecciones de choque/caída
  delay(50);
}

// Fija la velocidad (mm/s, rango -500..500) de cada rueda de forma directa
void setWheelSpeeds(int16_t left_mms, int16_t right_mms) {

  // Compensa la asimetria mecanica medida por calibrateWheelTrim()
  right_mms = (int16_t)(right_mms * wheelTrimFactor);

  left_mms = constrain(left_mms, -500, 500);
  right_mms = constrain(right_mms, -500, 500);

  // Si el robot está parado por un choque, ignorar nuevas velocidades
  // hasta que se libere explícitamente con clearBumpStop()
  if (sensors.stoppedByBump) {
    left_mms = 0;
    right_mms = 0;
  }

  targetLeftSpeed_mms = left_mms;
  targetRightSpeed_mms = right_mms;

  uint8_t cmd[5];
  cmd[0] = OI_DRIVE_DIRECT;
  cmd[1] = (uint8_t)(right_mms >> 8);
  cmd[2] = (uint8_t)(right_mms & 0xFF);
  cmd[3] = (uint8_t)(left_mms >> 8);
  cmd[4] = (uint8_t)(left_mms & 0xFF);
  RoombaSerial.write(cmd, 5);
}

void stopRobot() {
  targetLeftSpeed_mms = 0;
  targetRightSpeed_mms = 0;
  uint8_t cmd[5] = { OI_DRIVE_DIRECT, 0, 0, 0, 0 };
  RoombaSerial.write(cmd, 5);
}

// Permite volver a mover el robot tras una parada por choque
void clearBumpStop() {
  sensors.stoppedByBump = false;
  digitalWrite(BUMPER_LED_PIN, LOW);
}

// Pide el paquete de sensores configurado y actualiza el estado global
void pollSensors() {

  while (RoombaSerial.available()) {
    RoombaSerial.read();  // descarta restos de una lectura previa incompleta
    yield();
  }

  uint8_t req[2 + SENSOR_PACKETS_COUNT];
  req[0] = OI_QUERY_LIST;
  req[1] = SENSOR_PACKETS_COUNT;
  memcpy(&req[2], SENSOR_PACKETS, SENSOR_PACKETS_COUNT);
  RoombaSerial.write(req, sizeof(req));

  uint8_t buf[SENSOR_RESPONSE_BYTES];
  unsigned long start = millis();
  int received = 0;
  while (received < SENSOR_RESPONSE_BYTES && (millis() - start) < 100) {
    if (RoombaSerial.available()) {
      buf[received++] = RoombaSerial.read();
    }
  }
  if (received < SENSOR_RESPONSE_BYTES) {
    return;  // lectura incompleta: se reintenta en el próximo ciclo
  }

  int idx = 0;

  // --- Paquete 7: Bumps and Wheel Drops ---
  uint8_t bumpByte = buf[idx++];
  sensors.bumpRight = bumpByte & 0x01;
  sensors.bumpLeft = bumpByte & 0x02;
  sensors.wheelDropRight = bumpByte & 0x04;
  sensors.wheelDropLeft = bumpByte & 0x08;
  sensors.wheelDropCaster = bumpByte & 0x10;

  // --- Paquetes 43/44: Encoders (16 bits, con signo) ---
  int16_t leftEncoder = (int16_t)((buf[idx] << 8) | buf[idx + 1]);
  idx += 2;
  int16_t rightEncoder = (int16_t)((buf[idx] << 8) | buf[idx + 1]);
  idx += 2;

  // --- Paquete 22: Voltage (mV) ---
  uint16_t voltage_mv = (uint16_t)((buf[idx] << 8) | buf[idx + 1]);
  idx += 2;
  sensors.voltage_v = voltage_mv / 1000.0;

  // --- Paquete 25: Battery Charge (mAh) ---
  sensors.battery_charge_mah = (uint16_t)((buf[idx] << 8) | buf[idx + 1]);
  idx += 2;

  // --- Paquete 26: Battery Capacity (mAh) ---
  sensors.battery_capacity_mah = (uint16_t)((buf[idx] << 8) | buf[idx + 1]);
  idx += 2;

  if (sensors.battery_capacity_mah > 0) {
    sensors.battery_pct = 100.0 * sensors.battery_charge_mah / sensors.battery_capacity_mah;
  }

  updateOdometry(leftEncoder, rightEncoder);
  handleBumperSafety();
}

// ==================================================================
//  ODOMETRÍA
// ==================================================================

void updateOdometry(int16_t leftEncoder, int16_t rightEncoder) {
  if (firstEncoderReading) {
    lastLeftEncoder = leftEncoder;
    lastRightEncoder = rightEncoder;
    firstEncoderReading = false;
    return;
  }

  // La resta en int16_t maneja correctamente el desbordamiento (wrap-around)
  // del contador de 16 bits del Roomba
  int16_t deltaLeftCounts = (int16_t)(leftEncoder - lastLeftEncoder);
  int16_t deltaRightCounts = (int16_t)(rightEncoder - lastRightEncoder);
  lastLeftEncoder = leftEncoder;
  lastRightEncoder = rightEncoder;

  float deltaLeft_mm = deltaLeftCounts * MM_PER_COUNT;
  float deltaRight_mm = deltaRightCounts * MM_PER_COUNT;

  float deltaCenter_mm = (deltaLeft_mm + deltaRight_mm) / 2.0;
  float deltaTheta_rad = (deltaRight_mm - deltaLeft_mm) / WHEEL_BASE_MM;

  // Integración con corrección de punto medio (más precisa que Euler simple)
  float thetaMid = pose.theta_rad + deltaTheta_rad / 2.0;
  pose.x_mm += deltaCenter_mm * cos(thetaMid);
  pose.y_mm += deltaCenter_mm * sin(thetaMid);
  pose.theta_rad += deltaTheta_rad;

  // Normalizar theta a [-pi, pi]
  while (pose.theta_rad > PI) pose.theta_rad -= 2 * PI;
  while (pose.theta_rad < -PI) pose.theta_rad += 2 * PI;
}

Pose getPose() {
  return pose;
}

// ==================================================================
//  SEGURIDAD - PARADA POR CONTACTO FÍSICO
// ==================================================================

void handleBumperSafety() {
  if ((sensors.bumpLeft || sensors.bumpRight) && !sensors.stoppedByBump) {
    sensors.stoppedByBump = true;
    stopRobot();
    digitalWrite(BUMPER_LED_PIN, HIGH);
  }
}

// ==================================================================
//  CALIBRACIÓN DE ASIMETRÍA ENTRE RUEDAS
// ==================================================================

// Mueve ambas ruedas rectas (sin trim) durante unos segundos, compara
// cuanto giro realmente cada encoder, y ajusta wheelTrimFactor para que
// un mismo comando produzca el mismo avance fisico en ambas ruedas.
// BLOQUEANTE (~3.5s). Pensado para llamarse una sola vez, con el robot
// en espacio libre y suelo firme (no alfombra, para no falsear la medida).
void calibrateWheelTrim() {
  const int16_t testSpeed = 200;
  const unsigned long testDuration_ms = 3000;

  while (RoombaSerial.available()) RoombaSerial.read();
  uint8_t req[4] = { OI_QUERY_LIST, 2, 43, 44 };
  RoombaSerial.write(req, sizeof(req));

  uint8_t buf[4];
  int received = 0;
  unsigned long start = millis();
  while (received < 4 && (millis() - start) < 200) {
    if (RoombaSerial.available()) buf[received++] = RoombaSerial.read();
    yield();
  }
  if (received < 4) return;  // no se pudo leer, se mantiene el trim anterior

  int16_t leftStart = (int16_t)((buf[0] << 8) | buf[1]);
  int16_t rightStart = (int16_t)((buf[2] << 8) | buf[3]);

  // Comando crudo (sin trim), igual en ambas ruedas
  uint8_t drive[5] = {
    OI_DRIVE_DIRECT,
    (uint8_t)(testSpeed >> 8), (uint8_t)(testSpeed & 0xFF),
    (uint8_t)(testSpeed >> 8), (uint8_t)(testSpeed & 0xFF)
  };
  RoombaSerial.write(drive, 5);

  unsigned long moveStart = millis();
  while (millis() - moveStart < testDuration_ms) {
    yield();
  }

  uint8_t stopCmd[5] = { OI_DRIVE_DIRECT, 0, 0, 0, 0 };
  RoombaSerial.write(stopCmd, 5);

  while (RoombaSerial.available()) RoombaSerial.read();
  RoombaSerial.write(req, sizeof(req));
  received = 0;
  start = millis();
  while (received < 4 && (millis() - start) < 200) {
    if (RoombaSerial.available()) buf[received++] = RoombaSerial.read();
    yield();
  }
  if (received < 4) return;

  int16_t leftEnd = (int16_t)((buf[0] << 8) | buf[1]);
  int16_t rightEnd = (int16_t)((buf[2] << 8) | buf[3]);

  int16_t deltaLeft = (int16_t)(leftEnd - leftStart);
  int16_t deltaRight = (int16_t)(rightEnd - rightStart);

  if (deltaRight != 0) {
    wheelTrimFactor = (float)deltaLeft / (float)deltaRight;
  }
}

// ==================================================================
//  APAGADO DEL ROOMBA Y DEL ESP32
// ==================================================================

// ==================================================================
//  APAGADO DEL ROOMBA Y DEL ESP32
// ==================================================================

void powerDownRoomba() {
  RoombaSerial.write(133);  // opcode Power: equivale a mantener pulsado el boton "Clean" para apagar
}

void goToDeepSleep() {
  // El ESP32 queda en consumo minimo (decenas de uA). Sin una fuente de
  // wake-up configurada, solo se reactiva presionando el boton EN/reset.
  // Si mas adelante quieres reactivarlo remotamente, configura antes un
  // wake-up por temporizador (esp_sleep_enable_timer_wakeup) o por pin
  // externo (esp_sleep_enable_ext0_wakeup).
  esp_deep_sleep_start();
}

// ==================================================================
//  API HTTP (JSON) PARA LA PÁGINA WEB
// ==================================================================

void sendJson(int code, const String& body) {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(code, "application/json", body);
}

void handleGetState() {
  StaticJsonDocument<512> doc;
  doc["pose"]["x_mm"] = pose.x_mm;
  doc["pose"]["y_mm"] = pose.y_mm;
  doc["pose"]["theta_rad"] = pose.theta_rad;
  doc["pose"]["theta_deg"] = pose.theta_rad * 180.0 / PI;

  doc["sensors"]["bump_left"] = sensors.bumpLeft;
  doc["sensors"]["bump_right"] = sensors.bumpRight;
  doc["sensors"]["wheel_drop_left"] = sensors.wheelDropLeft;
  doc["sensors"]["wheel_drop_right"] = sensors.wheelDropRight;
  doc["sensors"]["wheel_drop_caster"] = sensors.wheelDropCaster;
  doc["sensors"]["voltage_v"] = sensors.voltage_v;
  doc["sensors"]["battery_pct"] = sensors.battery_pct;
  doc["sensors"]["stopped_by_bump"] = sensors.stoppedByBump;

  doc["speed"]["left_mms"] = targetLeftSpeed_mms;
  doc["speed"]["right_mms"] = targetRightSpeed_mms;
  doc["speed"]["wheel_trim_factor"] = wheelTrimFactor;

  String out;
  serializeJson(doc, out);
  sendJson(200, out);
}

void handleSetSpeed() {
  if (!server.hasArg("plain")) {
    sendJson(400, "{\"error\":\"cuerpo JSON requerido\"}");
    return;
  }
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, server.arg("plain"));
  if (err) {
    sendJson(400, "{\"error\":\"JSON invalido\"}");
    return;
  }
  int16_t left = doc["left"] | 0;
  int16_t right = doc["right"] | 0;
  setWheelSpeeds(left, right);
  sendJson(200, "{\"ok\":true}");
}

void handleStop() {
  stopRobot();
  sendJson(200, "{\"ok\":true}");
}

void handleClearBump() {
  clearBumpStop();
  sendJson(200, "{\"ok\":true}");
}

void handlePowerDown() {
  // Responder ANTES de dormir, para que el navegador reciba confirmacion
  sendJson(200, "{\"ok\":true,\"info\":\"apagando Roomba y entrando en deep sleep\"}");
  server.client().flush();
  powerDownRoomba();
  delay(500);  // tiempo para que el Roomba procese el comando de apagado
  goToDeepSleep();
}

void handleCalibrate() {
  calibrateWheelTrim();
  char resp[96];
  snprintf(resp, sizeof(resp), "{\"ok\":true,\"wheel_trim_factor\":%.4f}", wheelTrimFactor);
  sendJson(200, resp);
}

void setupWebServer() {
  server.on("/api/state", HTTP_GET, handleGetState);
  server.on("/api/speed", HTTP_POST, handleSetSpeed);
  server.on("/api/stop", HTTP_POST, handleStop);
  server.on("/api/clear_bump", HTTP_POST, handleClearBump);
  server.on("/api/power_down", HTTP_POST, handlePowerDown);
  server.on("/api/calibrate", HTTP_POST, handleCalibrate);
  server.begin();
}

// ==================================================================
//  SETUP / LOOP
// ==================================================================

void setup() {
  Serial.begin(115200);
  pinMode(BUMPER_LED_PIN, OUTPUT);
  digitalWrite(BUMPER_LED_PIN, LOW);

  RoombaSerial.begin(115200, SERIAL_8N1, ROOMBA_RX_PIN, ROOMBA_TX_PIN);

  wakeRoomba();
  delay(200);
  roombaStart();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Conectando a WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("IP del ESP32: ");
  Serial.println(WiFi.localIP());

  setupWebServer();
}

void loop() {
  server.handleClient();

  unsigned long now = millis();
  if (now - lastSensorPoll >= SENSOR_POLL_INTERVAL_MS) {
    lastSensorPoll = now;
    pollSensors();
  }
}
