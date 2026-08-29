/*
  ESP32-CAM Obstacle Detector — point-sampling + HSV comparison approach

  v4 REWRITE: replaced the full-frame color-mask approach with point
  sampling, per your spec:
    - N detection points, evenly spaced in a row in the image's upper region.
    - 3 reference ("floor") points, evenly spaced in a row in the lower region.
    - Each point is sampled as the average RGB of a 9x9 box around it.
    - Each of the N detection points' box-average is individually converted
      to HSV -> an array of N HSV values.
    - The 3 reference points' box-averages are averaged together *first* (in
      RGB), then that single combined value is converted to HSV -> one
      reference HSV.
    - Comparison uses H and S only (V/brightness ignored), with H treated as
      circular (hue wraps at 360 degrees).
    - A filled circle is drawn on the color output image at every detection
      point currently flagged as an obstacle; this color image (not a
      black/white mask) is what gets streamed.

  REUSED FROM THE PREVIOUS VERSION:
    - WiFi setup, JPEG capture (fast hardware-compressed sensor readout),
      fmt2rgb888 decode, fmt2jpg re-encode, and the MJPEG streaming loop.
    - The DRAM frame-buffer / pin config / camera setup are unchanged.

  DROPPED FROM THE PREVIOUS VERSION:
    - The full-frame color mask and the 5x5 box blur — no longer needed,
      since each point already gets its own local averaging (the 9x9 box
      IS the smoothing for that point).

  Requires: "esp32" board package (Espressif) in Arduino IDE.
  Board: AI Thinker ESP32-CAM (or similar).
*/

#include "esp_camera.h"
#include "img_converters.h"   // fmt2rgb888, fmt2jpg
#include <WiFi.h>
#include <WebServer.h>
#include <math.h>

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

// ---------- Point-sampling configuration ----------
#define N_POINTS 8            // number of detection points across the top row
const int BOX_RADIUS = 4;     // 9x9 box -> +/-4 pixels around the point
const int CIRCLE_RADIUS = 4;  // marker circle radius drawn on obstacles

// H is stored as 0-255 representing 0-360 degrees (circular).
// S and V are stored as 0-255 (standard 8-bit scale).
const uint8_t H_TOLERANCE = 30;   // tune: max circular hue difference still considered "floor"
const uint8_t S_TOLERANCE = 30;   // tune: max saturation difference still considered "floor"

struct Point { int x, y; };
Point detectPoints[N_POINTS];
Point refPoints[3];

struct HSV { uint8_t h, s, v; };

void initPoints() {
  // Detection points: evenly spaced row in the upper region of the frame.
  int marginX = FRAME_W / (N_POINTS + 1);
  int detectY = (int)(FRAME_H * 0.25f);
  for (int i = 0; i < N_POINTS; i++) {
    detectPoints[i].x = marginX * (i + 1);
    detectPoints[i].y = detectY;
  }

  // Reference (floor) points: 3 points evenly spaced in the lower region.
  int refY = (int)(FRAME_H * 0.85f);
  refPoints[0] = { (int)(FRAME_W * 0.25f), refY };
  refPoints[1] = { (int)(FRAME_W * 0.50f), refY };
  refPoints[2] = { (int)(FRAME_W * 0.75f), refY };
}

inline int clampCoord(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

// Average the RGB pixels in a 9x9 box (BOX_RADIUS=4 -> 9 wide) around (px,py).
void sampleBox9x9(uint8_t *rgb, int w, int h, int px, int py,
                   uint8_t &outR, uint8_t &outG, uint8_t &outB) {
  int sumR = 0, sumG = 0, sumB = 0, count = 0;
  for (int dy = -BOX_RADIUS; dy <= BOX_RADIUS; dy++) {
    int yy = clampCoord(py + dy, 0, h - 1);
    for (int dx = -BOX_RADIUS; dx <= BOX_RADIUS; dx++) {
      int xx = clampCoord(px + dx, 0, w - 1);
      int idx = (yy * w + xx) * 3;
      sumR += rgb[idx + 0];
      sumG += rgb[idx + 1];
      sumB += rgb[idx + 2];
      count++;
    }
  }
  outR = sumR / count;
  outG = sumG / count;
  outB = sumB / count;
}

// Standard RGB->HSV conversion. H,S,V all output as 0-255 (H represents
// 0-360 degrees mapped onto that range). Only called a handful of times per
// frame (N + 1), so float math here is cheap — no need to optimize this part.
HSV rgbToHsv(uint8_t r, uint8_t g, uint8_t b) {
  HSV out;
  uint8_t maxc = max(r, max(g, b));
  uint8_t minc = min(r, min(g, b));
  out.v = maxc;

  int delta = maxc - minc;
  if (maxc == 0 || delta == 0) {
    out.s = 0;
    out.h = 0;
    return out;
  }
  out.s = (uint8_t)((delta * 255) / maxc);

  float hf;
  if (maxc == r)      hf = 60.0f * fmodf(((float)(g - b) / delta), 6.0f);
  else if (maxc == g) hf = 60.0f * (((float)(b - r) / delta) + 2.0f);
  else                hf = 60.0f * (((float)(r - g) / delta) + 4.0f);
  if (hf < 0) hf += 360.0f;

  out.h = (uint8_t)(hf * 255.0f / 360.0f);
  return out;
}

// Circular hue difference: since H wraps at 256 (representing 360 degrees),
// a direct subtraction would be wrong near the wraparound point.
inline uint8_t hueDiff(uint8_t h1, uint8_t h2) {
  int d = abs((int)h1 - (int)h2);
  return (uint8_t)(d > 128 ? 256 - d : d);
}

void drawCircle(uint8_t *rgb, int w, int h, int cx, int cy, int radius,
                 uint8_t r, uint8_t g, uint8_t b) {
  for (int dy = -radius; dy <= radius; dy++) {
    int yy = cy + dy;
    if (yy < 0 || yy >= h) continue;
    for (int dx = -radius; dx <= radius; dx++) {
      int xx = cx + dx;
      if (xx < 0 || xx >= w) continue;
      if (dx * dx + dy * dy <= radius * radius) {
        int idx = (yy * w + xx) * 3;
        rgb[idx + 0] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
      }
    }
  }
}

// Samples all points, builds the reference HSV, compares each detection
// point against it (H and S only), and draws a circle on obstacle points
// directly onto the color image passed in.
void detectAndMark(uint8_t *rgb, int w, int h) {
  // --- Reference: sample 3 points, average their RGB together FIRST,
  // then convert that single combined average to HSV. ---
  uint8_t r0, g0, b0, r1, g1, b1, r2, g2, b2;
  sampleBox9x9(rgb, w, h, refPoints[0].x, refPoints[0].y, r0, g0, b0);
  sampleBox9x9(rgb, w, h, refPoints[1].x, refPoints[1].y, r1, g1, b1);
  sampleBox9x9(rgb, w, h, refPoints[2].x, refPoints[2].y, r2, g2, b2);

  uint8_t refR = (uint8_t)(((int)r0 + r1 + r2) / 3);
  uint8_t refG = (uint8_t)(((int)g0 + g1 + g2) / 3);
  uint8_t refB = (uint8_t)(((int)b0 + b1 + b2) / 3);
  HSV refHsv = rgbToHsv(refR, refG, refB);

  // --- Detection points: each sampled and converted to HSV independently. ---
  for (int i = 0; i < N_POINTS; i++) {
    uint8_t pr, pg, pb;
    sampleBox9x9(rgb, w, h, detectPoints[i].x, detectPoints[i].y, pr, pg, pb);
    HSV pointHsv = rgbToHsv(pr, pg, pb);

    uint8_t dh = hueDiff(pointHsv.h, refHsv.h);
    uint8_t ds = (uint8_t)abs((int)pointHsv.s - (int)refHsv.s);

    bool isObstacle = (dh > H_TOLERANCE) || (ds > S_TOLERANCE);
    if (isObstacle) {
      drawCircle(rgb, w, h, detectPoints[i].x, detectPoints[i].y, CIRCLE_RADIUS,
                 255, 0, 0);  // red marker
    }
  }
}

void handleStream() {
  WiFiClient client = server.client();
  String boundary = "frame";
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=" + boundary + "\r\n\r\n";
  server.sendContent(response);

  static uint8_t *rgbBuf = (uint8_t *)malloc(FRAME_W * FRAME_H * 3);
  if (!rgbBuf) {
    Serial.println("Failed to allocate rgbBuf");
    return;
  }

  while (client.connected()) {
    uint32_t t0 = micros();
    camera_fb_t *fb = esp_camera_fb_get();   // JPEG-compressed frame, fast sensor readout
    if (!fb) continue;
    uint32_t t1 = micros();

    bool decoded = fmt2rgb888(fb->buf, fb->len, fb->format, rgbBuf);
    uint32_t t2 = micros();

    esp_camera_fb_return(fb);   // done with the JPEG source buffer

    if (decoded) {
      detectAndMark(rgbBuf, FRAME_W, FRAME_H);
    }
    uint32_t t3 = micros();

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
    static uint32_t capUs = 0, decUs = 0, detUs = 0, encUs = 0, sendUs = 0, frames = 0;
    capUs  += (t1 - t0);
    decUs  += (t2 - t1);
    detUs  += (t3 - t2);
    encUs  += (t4 - t3);
    sendUs += (t5 - t4);
    frames++;
    if (millis() - lastReport > 1000) {
      Serial.printf("FPS:%d  capture:%lums  decode:%lums  detect:%lums  jpegEnc:%lums  send:%lums\n",
        frames,
        (unsigned long)(capUs / frames / 1000),
        (unsigned long)(decUs / frames / 1000),
        (unsigned long)(detUs / frames / 1000),
        (unsigned long)(encUs / frames / 1000),
        (unsigned long)(sendUs / frames / 1000));
      capUs = decUs = detUs = encUs = sendUs = 0;
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

  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QQVGA;   // 160x120
  config.jpeg_quality = 12;
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
  initPoints();

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
  - N_POINTS, BOX_RADIUS, CIRCLE_RADIUS: adjust point count/coverage and
    marker size to taste.
  - H_TOLERANCE / S_TOLERANCE: tune independently, same philosophy as the
    earlier per-channel discussion — if you're getting false positives on
    normal floor texture, widen the relevant tolerance; if you're missing
    real obstacles, narrow it.
  - detectPoints/refPoints are currently a single row each. If your floor
    has perspective (near vs far parts of the floor look different), you
    may want the detection row higher/lower, or multiple rows — easy to
    extend inside initPoints().
  - This point-sampling approach is inherently much cheaper than the old
    full-frame mask (only N+3 box samples and N+1 HSV conversions per frame,
    versus touching every pixel), so "detect" time in the timing report
    should be small — worth checking against the earlier full-frame numbers.
*/
