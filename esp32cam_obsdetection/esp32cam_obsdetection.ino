/*
  ESP32-CAM Obstacle Detector — floor-color-difference approach
  Equivalent to the Python/OpenCV/Flask script, adapted for ESP32 constraints.

  KEY DIFFERENCES FROM THE PYTHON VERSION (and why):
  - No OpenCV (won't fit/run well on ESP32). We do direct pixel math on the
    raw framebuffer instead.
  - We work in RGB565 (the camera's native low-RAM format) rather than
    converting every frame to HSV — a full HSV conversion per-pixel per-frame
    is too expensive for the ESP32's CPU/RAM budget at any decent frame rate.
    Instead we use a simple Euclidean color-distance test against a sampled
    "floor color", which is cheap and works well enough for this use case.
  - No percentile calculation (np.percentile) — we use a fast running
    min/max/mean sample of a floor patch every N frames instead of a full
    sort/percentile, since sorting per pixel-channel on ESP32 is wasteful.
  - Uses the AI-Thinker ESP32-CAM board pinout. Adjust CAMERA_MODEL below
    if you're using a different board.

  Requires: "esp32" board package (Espressif) in Arduino IDE, with the
  built-in esp_camera library. Board: AI Thinker ESP32-CAM (or similar).
*/

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ---------- WiFi credentials ----------
const char* ssid     = "Sarita";
const char* password = "bubududu";

// ---------- AI-Thinker ESP32-CAM pin map ----------
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

WebServer server(80);

// Sampled floor color (in RGB565-derived 8-bit RGB) and tolerance
uint8_t floorR = 128, floorG = 128, floorB = 128;
const int COLOR_TOLERANCE = 35;   // Euclidean-ish distance threshold, tune this
const int RESAMPLE_EVERY_N_FRAMES = 10;
uint32_t frameCount = 0;

// Convert a RGB565 pixel to 8-bit R,G,B components
inline void rgb565_to_rgb888(uint16_t p, uint8_t &r, uint8_t &g, uint8_t &b) {
  r = (p >> 11) & 0x1F; r = (r * 255) / 31;
  g = (p >> 5)  & 0x3F; g = (g * 255) / 63;
  b =  p        & 0x1F; b = (b * 255) / 31;
}

void resampleFloorColor(camera_fb_t *fb) {
  // Sample a patch in the bottom-center of the frame (like the Python version's
  // hsv[0.8h:h, 0.3w:0.7w] region), average it as the reference floor color.
  uint16_t *pixels = (uint16_t *)fb->buf;
  int w = fb->width, h = fb->height;

  int y0 = (int)(h * 0.8f), y1 = h;
  int x0 = (int)(w * 0.3f), x1 = (int)(w * 0.7f);

  uint32_t sumR = 0, sumG = 0, sumB = 0, count = 0;
  for (int y = y0; y < y1; y += 2) {         // step by 2 to save CPU
    for (int x = x0; x < x1; x += 2) {
      uint16_t p = pixels[y * w + x];
      uint8_t r, g, b;
      rgb565_to_rgb888(p, r, g, b);
      sumR += r; sumG += g; sumB += b;
      count++;
    }
  }
  if (count > 0) {
    floorR = sumR / count;
    floorG = sumG / count;
    floorB = sumB / count;
  }
}

// Builds an obstacle mask directly into a JPEG-ready buffer by overwriting
// framebuffer pixels: floor -> black, obstacle -> white. This mimics the
// "mask" half of your side-by-side view, but cheaper: we skip the hstack and
// just output the mask (stream both if you have RAM/bandwidth to spare).
void computeObstacleMaskInPlace(camera_fb_t *fb) {
  uint16_t *pixels = (uint16_t *)fb->buf;
  int total = fb->width * fb->height;

  for (int i = 0; i < total; i++) {
    uint8_t r, g, b;
    rgb565_to_rgb888(pixels[i], r, g, b);

    int dr = (int)r - floorR;
    int dg = (int)g - floorG;
    int db = (int)b - floorB;
    // squared distance avoids a sqrt() call per pixel
    int distSq = dr*dr + dg*dg + db*db;
    int thresholdSq = COLOR_TOLERANCE * COLOR_TOLERANCE;

    if (distSq > thresholdSq) {
      pixels[i] = 0xFFFF; // obstacle -> white
    } else {
      pixels[i] = 0x0000; // floor -> black
    }
  }
}

void handleStream() {
  WiFiClient client = server.client();
  String boundary = "frame";
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=" + boundary + "\r\n\r\n";
  server.sendContent(response);

  while (client.connected()) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) continue;

    if (frameCount % RESAMPLE_EVERY_N_FRAMES == 0) {
      resampleFloorColor(fb);
    }
    computeObstacleMaskInPlace(fb);   // mutates fb->buf in place (RGB565 mask)
    frameCount++;

    // Re-encode the RGB565 mask buffer as JPEG for transport (esp_camera can
    // give us JPEG directly if you configure PIXFORMAT_JPEG instead — see note below)
    uint8_t *jpg_buf = NULL;
    size_t jpg_len = 0;
    bool converted = frame2jpg(fb, 60, &jpg_buf, &jpg_len);

    if (converted) {
      client.printf("--%s\r\n", boundary.c_str());
      client.printf("Content-Type: image/jpeg\r\n\r\n");
      client.write(jpg_buf, jpg_len);
      client.print("\r\n");
      free(jpg_buf);
    }

    esp_camera_fb_return(fb);
    if (!client.connected()) break;
  }
}

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  //config.fb_location = CAMERA_FB_IN_DRAM;

  // RGB565 so we can do pixel-level color math directly (like the Python
  // script's HSV per-pixel access). Keep resolution LOW — this is the single
  // biggest lever for staying real-time on ESP32.
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QQVGA;   // 160x120 — start here, raise only if it stays smooth
  config.jpeg_quality = 12;              // unused for RGB565 capture, kept for reference
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    while (true) delay(1000);
  }
}

void setup() {
  Serial.begin(115200);
  setupCamera();

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Camera stream ready at: http://");
  Serial.println(WiFi.localIP());

  server.on("/", HTTP_GET, handleStream);
  server.begin();
}

void loop() {
  server.handleClient();
}

/*
  TUNING NOTES:
  - FRAMESIZE_QQVGA (160x120) is a deliberately small starting point. The
    Python script used 320x240; on ESP32 that resolution in RGB565 with
    per-pixel processing will likely choke your frame rate. Step up only
    after confirming QQVGA runs smoothly.
  - COLOR_TOLERANCE plays the role of your min_hsv/max_hsv percentile band.
    Since we're not in HSV, lighting changes will affect this more than they
    did in the Python version (HSV separates brightness from color; raw RGB
    doesn't). If false positives from shadows/glare are a problem, consider
    normalizing brightness first (e.g., divide each channel by their sum) or
    porting a lightweight RGB->HSV conversion (it's roughly 15-20 extra ops
    per pixel — still much cheaper than OpenCV, but not free).
  - Sampling every 2nd pixel in resampleFloorColor() and using squared
    distance instead of sqrt() are the two cheapest wins for keeping this
    real-time on ESP32's single relevant core.
*/
