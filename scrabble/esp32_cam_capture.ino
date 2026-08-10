/*
  ESP32-CAM - Captura periódica del tablero de Scrabble
  ------------------------------------------------------
  Placa: AI-Thinker ESP32-CAM
  Función: conectarse al WiFi y enviar una foto por HTTP POST
           al servidor Python cada CAPTURE_INTERVAL_MS.

  Cableado para grabar el firmware (AI-Thinker):
    - GPIO0 a GND durante el reset para entrar en modo flash.
    - Quitar el puente GPIO0-GND y resetear para ejecutar el programa.

  Librerías necesarias (Arduino IDE / Board Manager):
    - "esp32" by Espressif Systems (Board Manager)
    - No hace falta librería extra de cámara: se usa esp_camera.h que
      viene incluida con el core de ESP32.
*/

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ---------- CONFIGURACIÓN DEL USUARIO ----------
const char* WIFI_SSID     = "TU_WIFI";
const char* WIFI_PASSWORD = "TU_PASSWORD";

// IP y puerto donde corre server.py (tu PC en la misma red WiFi)
const char* SERVER_URL     = "http://192.168.1.100:5000/upload";
const char* NEXT_TURN_URL  = "http://192.168.1.100:5000/next_turn";

// Cada cuánto se envía una foto (ms). 3000 = cada 3 segundos.
const unsigned long CAPTURE_INTERVAL_MS = 3000;

// ---------- BOTÓN FÍSICO DE CAMBIO DE TURNO ----------
// Conectar un pulsador entre GPIO13 y GND (usa la resistencia pull-up
// interna, así que no hace falta resistencia externa).
// GPIO13 no se usa para la cámara ni es pin de arranque, por eso es seguro.
#define BUTTON_PIN 13
const unsigned long DEBOUNCE_MS = 250;
unsigned long lastButtonPress = 0;

// ---------- PINES CÁMARA AI-THINKER ----------
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

unsigned long lastCapture = 0;

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Conectando a WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Conectado. IP: ");
  Serial.println(WiFi.localIP());
}

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // SVGA (800x600) es un buen compromiso resolución/velocidad para
  // luego recortar 225 casillas con detalle suficiente para reconocer letras.
  if (psramFound()) {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 10; // menor = más calidad
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Fallo al iniciar la cámara: 0x%x\n", err);
    return false;
  }
  return true;
}

void sendFrame() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Error al capturar frame");
    return;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "image/jpeg");
  int httpCode = http.POST(fb->buf, fb->len);

  if (httpCode > 0) {
    Serial.printf("Foto enviada. Respuesta servidor: %d\n", httpCode);
  } else {
    Serial.printf("Error enviando foto: %s\n", http.errorToString(httpCode).c_str());
  }

  http.end();
  esp_camera_fb_return(fb);
}

void notifyNextTurn() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Sin WiFi: no se pudo avisar del cambio de turno.");
    return;
  }
  HTTPClient http;
  http.begin(NEXT_TURN_URL);
  int httpCode = http.POST("");
  if (httpCode > 0) {
    Serial.printf("Cambio de turno notificado. Respuesta: %d\n", httpCode);
  } else {
    Serial.printf("Error notificando cambio de turno: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);

  pinMode(BUTTON_PIN, INPUT_PULLUP);

  if (!initCamera()) {
    Serial.println("No se pudo iniciar la cámara. Reiniciando...");
    delay(3000);
    ESP.restart();
  }

  connectWiFi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  unsigned long now = millis();

  // Botón pulsado = nivel LOW (pull-up interno), con anti-rebote simple.
  if (digitalRead(BUTTON_PIN) == LOW && (now - lastButtonPress) > DEBOUNCE_MS) {
    lastButtonPress = now;
    Serial.println("Botón de cambio de turno pulsado");
    notifyNextTurn();
  }

  if (now - lastCapture >= CAPTURE_INTERVAL_MS) {
    lastCapture = now;
    sendFrame();
  }
}
