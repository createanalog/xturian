/*
  ESP32-CAM Obstacle Detector — floor-color-difference approach
  Equivalent to the Python/OpenCV/Flask script, adapted for ESP32 constraints.

  v2 CHANGE: capture is now done in JPEG (hardware-compressed) instead of
  RGB565. The OV2640's raw/uncompressed capture path is known to be limited
  to roughly ~7 FPS at QQVGA regardless of lighting or code — that's a sensor
  pipeline limitation, not something fixable in software. JPEG capture uses
  the sensor's onboard hardware encoder and is typically 3-5x faster to read
  out, at the cost of needing to decode it back to raw pixels in software
  before running the color-distance mask (fmt2rgb888) and re-encode the
  result to JPEG for streaming (fmt2jpg). Both conversions use the esp32
  library's own lightweight (de)coders — no OpenCV needed.

  v3 CHANGE: added a separable 5x5 box-blur smoothing pass right after
  decode, before floor sampling and mask computation. Smooths out sensor
  noise/small speckling so the per-channel floor bounds test (see below)
  isn't thrown off by isolated noisy pixels.

  v3 CHANGE (also): switched from a single combined Euclidean color-distance
  threshold to independent per-channel [min,max] bounds (matching the
  original cv2.inRange per-channel behavior), since a floor can vary a lot
  in one channel while staying tight in others — a single combined tolerance
  forces a bad tradeoff between missing real obstacles and flagging natural
  floor texture as obstacles.

  KEY DIFFERENCES FROM THE ORIGINAL PYTHON VERSION:
  - No OpenCV. Direct pixel math on a decoded RGB888 buffer.
  - No full HSV conversion — uses Euclidean color-distance from a sampled
    floor color instead of HSV min/max percentile thresholds.
  - No percentile calc — uses an averaged sample of a floor patch instead.
  - Capture format is JPEG (fast sensor readout), decoded to RGB888 in
    software for processing, then re-encoded to JPEG for the MJPEG stream.

  Requires: "esp32" board package (Espressif) in Arduino IDE.
  Board: AI Thinker ESP32-CAM (or similar).
*/

#include "esp_camera.h"
#include "img_converters.h"   // fmt2rgb888, fmt2jpg
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

const int FRAME_W = 160;
const int FRAME_H = 120;

// Per-channel bounds instead of a single combined distance — this matches
// the original cv2.inRange() behavior (an axis-aligned box in color space)
// rather than a sphere around one average color. Important because a floor
// can vary a lot in one channel (e.g. red, from wood grain/lighting) while
// staying tight in others — a single combined tolerance forces a bad
// tradeoff between missing real obstacles and flagging natural texture as
// obstacles. Independent per-channel bounds avoid that tradeoff.
uint8_t floorRmin = 100, floorRmax = 156;
uint8_t floorGmin = 100, floorGmax = 156;
uint8_t floorBmin = 100, floorBmax = 156;
const int CHANNEL_MARGIN = 10;   // padding added beyond observed min/max, tune this
const int RESAMPLE_EVERY_N_FRAMES = 10;
uint32_t frameCount = 0;

void resampleFloorColor(uint8_t *rgb, int w, int h) {
  // Sample a patch in the bottom-center of the frame (like the Python
  // version's hsv[0.8h:h, 0.3w:0.7w] region), track per-channel min/max.
  int y0 = (int)(h * 0.8f), y1 = h;
  int x0 = (int)(w * 0.3f), x1 = (int)(w * 0.7f);

  uint8_t rMin = 255, rMax = 0, gMin = 255, gMax = 0, bMin = 255, bMax = 0;
  bool any = false;
  for (int y = y0; y < y1; y += 2) {
    for (int x = x0; x < x1; x += 2) {
      int idx = (y * w + x) * 3;
      uint8_t r = rgb[idx + 0], g = rgb[idx + 1], b = rgb[idx + 2];
      if (r < rMin) rMin = r;  if (r > rMax) rMax = r;
      if (g < gMin) gMin = g;  if (g > gMax) gMax = g;
      if (b < bMin) bMin = b;  if (b > bMax) bMax = b;
      any = true;
    }
  }
  if (any) {
    // NOTE: this uses true min/max (not the 5th/95th percentile your Python
    // version used), so a few noisy/outlier pixels can widen the band more
    // than percentiles would. If you see the mask getting too permissive,
    // that's why — consider averaging min/max across a couple of resamples,
    // or shrinking CHANNEL_MARGIN to compensate.
    floorRmin = (rMin > CHANNEL_MARGIN) ? rMin - CHANNEL_MARGIN : 0;
    floorRmax = (rMax + CHANNEL_MARGIN < 255) ? rMax + CHANNEL_MARGIN : 255;
    floorGmin = (gMin > CHANNEL_MARGIN) ? gMin - CHANNEL_MARGIN : 0;
    floorGmax = (gMax + CHANNEL_MARGIN < 255) ? gMax + CHANNEL_MARGIN : 255;
    floorBmin = (bMin > CHANNEL_MARGIN) ? bMin - CHANNEL_MARGIN : 0;
    floorBmax = (bMax + CHANNEL_MARGIN < 255) ? bMax + CHANNEL_MARGIN : 255;
  }
}

// Builds the obstacle mask in-place: a pixel counts as "floor" only if ALL
// THREE channels fall within their own independent bounds (matching
// cv2.inRange's per-channel AND behavior) — otherwise it's "obstacle".
void computeObstacleMaskInPlace(uint8_t *rgb, int w, int h) {
  int total = w * h;

  for (int i = 0; i < total; i++) {
    int idx = i * 3;
    uint8_t r = rgb[idx + 0], g = rgb[idx + 1], b = rgb[idx + 2];

    bool isFloor = (r >= floorRmin && r <= floorRmax) &&
                   (g >= floorGmin && g <= floorGmax) &&
                   (b >= floorBmin && b <= floorBmax);

    uint8_t v = isFloor ? 0 : 255;
    rgb[idx + 0] = v;
    rgb[idx + 1] = v;
    rgb[idx + 2] = v;
  }
}

// Separable 5x5 box blur (horizontal pass then vertical pass) — same result
// as a naive 5x5 blur but ~2.5x cheaper (2x5 samples/pixel/channel instead
// of 25), which matters on ESP32's limited CPU. Edge pixels clamp to the
// nearest valid coordinate (edge replication) rather than reading out of
// bounds. `tmp` and `dst` must both be w*h*3 buffers distinct from `src`
// (the vertical pass reads tmp and writes dst, so dst can safely be the
// same buffer as src — it's only read via tmp by that point).
inline int clampCoord(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

void boxBlur5x5(uint8_t *src, uint8_t *tmp, uint8_t *dst, int w, int h) {
  // Horizontal pass: src -> tmp
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int sumR = 0, sumG = 0, sumB = 0;
      for (int k = -2; k <= 2; k++) {
        int xx = clampCoord(x + k, 0, w - 1);
        int idx = (y * w + xx) * 3;
        sumR += src[idx + 0];
        sumG += src[idx + 1];
        sumB += src[idx + 2];
      }
      int outIdx = (y * w + x) * 3;
      tmp[outIdx + 0] = sumR / 5;
      tmp[outIdx + 1] = sumG / 5;
      tmp[outIdx + 2] = sumB / 5;
    }
  }

  // Vertical pass: tmp -> dst
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) {
      int sumR = 0, sumG = 0, sumB = 0;
      for (int k = -2; k <= 2; k++) {
        int yy = clampCoord(y + k, 0, h - 1);
        int idx = (yy * w + x) * 3;
        sumR += tmp[idx + 0];
        sumG += tmp[idx + 1];
        sumB += tmp[idx + 2];
      }
      int outIdx = (y * w + x) * 3;
      dst[outIdx + 0] = sumR / 5;
      dst[outIdx + 1] = sumG / 5;
      dst[outIdx + 2] = sumB / 5;
    }
  }
}

void handleStream() {
  WiFiClient client = server.client();
  String boundary = "frame";
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=" + boundary + "\r\n\r\n";
  server.sendContent(response);

  // Reusable RGB888 working buffer (allocated once, not per-frame)
  static uint8_t *rgbBuf = (uint8_t *)malloc(FRAME_W * FRAME_H * 3);
  static uint8_t *blurTmp = (uint8_t *)malloc(FRAME_W * FRAME_H * 3);
  if (!rgbBuf || !blurTmp) {
    Serial.println("Failed to allocate rgbBuf/blurTmp");
    return;
  }

  while (client.connected()) {
    uint32_t t0 = micros();
    camera_fb_t *fb = esp_camera_fb_get();   // JPEG-compressed frame, fast sensor readout
    if (!fb) continue;
    uint32_t t1 = micros();

    bool decoded = fmt2rgb888(fb->buf, fb->len, fb->format, rgbBuf);
    uint32_t t2 = micros();

    if (decoded) {
      boxBlur5x5(rgbBuf, blurTmp, rgbBuf, FRAME_W, FRAME_H);
    }
    uint32_t t2b = micros();

    if (decoded) {
      if (frameCount % RESAMPLE_EVERY_N_FRAMES == 0) {
        resampleFloorColor(rgbBuf, FRAME_W, FRAME_H);
      }
      computeObstacleMaskInPlace(rgbBuf, FRAME_W, FRAME_H);
    }
    frameCount++;
    uint32_t t3 = micros();

    esp_camera_fb_return(fb);   // done with the JPEG source buffer

    uint8_t *jpg_buf = NULL;
    size_t jpg_len = 0;
    bool converted = false;
    if (decoded) {
      converted = fmt2jpg(rgbBuf, FRAME_W * FRAME_H * 3, FRAME_W, FRAME_H,
                           PIXFORMAT_RGB888, 60, &jpg_buf, &jpg_len);
    }
    uint32_t t4 = micros();

    if (converted) {
      client.printf("--%s\r\n", boundary.c_str());
      client.printf("Content-Type: image/jpeg\r\n\r\n");
      client.write(jpg_buf, jpg_len);
      client.print("\r\n");
      free(jpg_buf);
    }
    uint32_t t5 = micros();

    if (!client.connected()) break;

    // --- Timing report: prints once a second. Remove once tuned. ---
    static uint32_t lastReport = 0;
    static uint32_t capUs = 0, decUs = 0, blurUs = 0, maskUs = 0, encUs = 0, sendUs = 0, frames = 0;
    capUs  += (t1 - t0);
    decUs  += (t2 - t1);
    blurUs += (t2b - t2);
    maskUs += (t3 - t2b);
    encUs  += (t4 - t3);
    sendUs += (t5 - t4);
    frames++;
    if (millis() - lastReport > 1000) {
      Serial.printf("FPS:%d  capture:%lums  decode:%lums  blur:%lums  mask:%lums  jpegEnc:%lums  send:%lums\n",
        frames,
        (unsigned long)(capUs / frames / 1000),
        (unsigned long)(decUs / frames / 1000),
        (unsigned long)(blurUs / frames / 1000),
        (unsigned long)(maskUs / frames / 1000),
        (unsigned long)(encUs / frames / 1000),
        (unsigned long)(sendUs / frames / 1000));
      capUs = decUs = blurUs = maskUs = encUs = sendUs = 0;
      frames = 0;
      lastReport = millis();
    }
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

  // JPEG capture: fast hardware-compressed sensor readout. We decode this
  // back to RGB888 in software (see handleStream) before running the mask.
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QQVGA;   // 160x120
  config.jpeg_quality = 12;              // capture quality; lower number = higher quality/larger
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_DRAM;

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
  - If capture (t1-t0) is still slow even in JPEG mode, try raising
    config.jpeg_quality's number (e.g. 20-30) for a smaller/faster-to-produce
    JPEG from the sensor, at the cost of more compression artifacts feeding
    into fmt2rgb888.
  - decode (fmt2rgb888) and re-encode (fmt2jpg) are now separate costs the
    RGB565 version didn't have — watch these in the Serial output. If they
    dominate, consider skipping the re-encode entirely: since the mask is
    binary black/white, you could send a raw bitmap or a tiny "obstacle
    direction" decision over a lightweight custom protocol instead of MJPEG,
    which is likely the better design once you're past debugging anyway.
  - COLOR_TOLERANCE and RESAMPLE_EVERY_N_FRAMES behave the same as before.
*/
