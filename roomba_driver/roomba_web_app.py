"""
App web (Flask) para monitorizar y controlar el robot Roomba+ESP32.

Actúa como proxy entre el navegador y el ESP32: evita problemas de CORS
y deja un punto único donde añadir lógica extra en el futuro (límites
de seguridad, registro de trayectoria, etc.).

Requisitos:
    pip install flask requests

Antes de ejecutar, cambia ESP32_IP por la IP que el ESP32 imprime por
el puerto serie al conectarse al WiFi (ver Serial.println(WiFi.localIP())
en el sketch).

Ejecución:
    python roomba_web_app.py
    -> abrir http://localhost:5000 en el navegador
"""

from flask import Flask, jsonify, request, render_template_string
import requests

app = Flask(__name__)

ESP32_IP = "192.168.1.132"  # <-- CAMBIAR por la IP real del ESP32
ESP32_BASE_URL = f"http://{ESP32_IP}"
REQUEST_TIMEOUT = 2  # segundos

PAGE_TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Control Roomba</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 30px auto; padding: 0 16px; background:#111; color:#eee;}
  h1 { font-size: 1.4rem; }
  .card { background:#1c1c1c; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
  .row { display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #2a2a2a;}
  .row:last-child{border-bottom:none;}
  .ok { color:#4caf50; }
  .bad { color:#e53935; font-weight:bold; }
  label { display:block; margin-bottom:4px; font-size:0.9rem; color:#aaa;}
  input[type=number]{ width:100%; padding:8px; border-radius:6px; border:1px solid #333; background:#0d0d0d; color:#eee; box-sizing:border-box;}
  .controls { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  button { padding:10px; border:none; border-radius:6px; font-size:1rem; cursor:pointer; margin-top:10px; width:100%;}
  #btnSet { background:#2196f3; color:white; }
  #btnStop { background:#e53935; color:white; }
  #btnClear { background:#555; color:white; }
  #status { font-size:0.8rem; color:#888; margin-top:8px;}
</style>
</head>
<body>
  <h1>Control del Roomba</h1>

  <div class="card">
    <h3>Posicion estimada</h3>
    <div class="row"><span>X (mm)</span><span id="x">-</span></div>
    <div class="row"><span>Y (mm)</span><span id="y">-</span></div>
    <div class="row"><span>Angulo (grados)</span><span id="theta">-</span></div>
  </div>

  <div class="card">
    <h3>Sensores</h3>
    <div class="row"><span>Bumper izquierdo</span><span id="bumpL">-</span></div>
    <div class="row"><span>Bumper derecho</span><span id="bumpR">-</span></div>
    <div class="row"><span>Bateria</span><span id="batt">-</span></div>
    <div class="row"><span>Voltaje</span><span id="volt">-</span></div>
    <div class="row"><span>Parado por choque</span><span id="stopped">-</span></div>
  </div>

  <div class="card">
    <h3>Velocidad de ruedas (mm/s, rango -500 a 500)</h3>
    <div class="controls">
      <div>
        <label for="left">Rueda izquierda</label>
        <input type="number" id="left" value="0" min="-500" max="500">
      </div>
      <div>
        <label for="right">Rueda derecha</label>
        <input type="number" id="right" value="0" min="-500" max="500">
      </div>
    </div>
    <button id="btnSet">Aplicar velocidades</button>
    <button id="btnStop">PARAR</button>
    <button id="btnClear">Reanudar tras choque</button>
    <div id="status"></div>
  </div>

  <div class="card">
    <h3>Apagado</h3>
    <button id="btnPowerDown" style="background:#8a4b00;color:white;">Apagar Roomba y ESP32</button>
    <div id="powerStatus" style="font-size:0.8rem;color:#888;margin-top:8px;"></div>
  </div>

  <div class="card">
    <h3>Calibracion de ruedas</h3>
    <p style="font-size:0.85rem;color:#aaa;">Mueve el robot recto ~3 segundos y ajusta el factor de correccion entre ruedas. Deja espacio libre delante.</p>
    <button id="btnCalibrate" style="background:#6a1b9a;color:white;">Calibrar ruedas</button>
    <div id="calibrateStatus" style="font-size:0.8rem;color:#888;margin-top:8px;"></div>
  </div>

<script>
async function refreshState() {
  try {
    const res = await fetch('/state');
    const data = await res.json();
    document.getElementById('x').textContent = data.pose.x_mm.toFixed(1);
    document.getElementById('y').textContent = data.pose.y_mm.toFixed(1);
    document.getElementById('theta').textContent = data.pose.theta_deg.toFixed(1);

    const bumpL = data.sensors.bump_left;
    const bumpR = data.sensors.bump_right;
    document.getElementById('bumpL').innerHTML = bumpL ? '<span class="bad">CONTACTO</span>' : '<span class="ok">libre</span>';
    document.getElementById('bumpR').innerHTML = bumpR ? '<span class="bad">CONTACTO</span>' : '<span class="ok">libre</span>';
    document.getElementById('batt').textContent = data.sensors.battery_pct.toFixed(0) + ' %';
    document.getElementById('volt').textContent = data.sensors.voltage_v.toFixed(2) + ' V';

    const stopped = data.sensors.stopped_by_bump;
    document.getElementById('stopped').innerHTML = stopped ? '<span class="bad">SI</span>' : '<span class="ok">no</span>';

    document.getElementById('status').textContent = 'Actualizado ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('status').textContent = 'Sin conexion con el ESP32';
  }
}

document.getElementById('btnSet').addEventListener('click', async () => {
  const left = parseInt(document.getElementById('left').value, 10) || 0;
  const right = parseInt(document.getElementById('right').value, 10) || 0;
  await fetch('/speed', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({left, right})
  });
});

document.getElementById('btnStop').addEventListener('click', async () => {
  document.getElementById('left').value = 0;
  document.getElementById('right').value = 0;
  await fetch('/stop', { method: 'POST' });
});

document.getElementById('btnClear').addEventListener('click', async () => {
  await fetch('/clear_bump', { method: 'POST' });
});

document.getElementById('btnPowerDown').addEventListener('click', async () => {
  const confirmado = confirm('Esto apaga el Roomba y pone el ESP32 en deep sleep. Para reactivarlo necesitaras pulsar el boton EN/reset de la placa. Continuar?');
  if (!confirmado) return;
  const statusEl = document.getElementById('powerStatus');
  try {
    await fetch('/power_down', { method: 'POST' });
    statusEl.textContent = 'Comando enviado. El robot se esta apagando...';
  } catch (e) {
    statusEl.textContent = 'No se pudo contactar al ESP32 (puede que ya se haya apagado).';
  }
});

document.getElementById('btnCalibrate').addEventListener('click', async () => {
  const confirmado = confirm('El robot avanzara recto durante unos 3 segundos. Asegurate de que tenga espacio libre delante. Continuar?');
  if (!confirmado) return;
  const statusEl = document.getElementById('calibrateStatus');
  statusEl.textContent = 'Calibrando...';
  try {
    const res = await fetch('/calibrate', { method: 'POST' });
    const data = await res.json();
    statusEl.textContent = 'Factor de correccion aplicado: ' + data.wheel_trim_factor.toFixed(4);
  } catch (e) {
    statusEl.textContent = 'No se pudo completar la calibracion.';
  }
});

setInterval(refreshState, 400);
refreshState();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE_TEMPLATE)


@app.route("/state")
def state():
    try:
        r = requests.get(f"{ESP32_BASE_URL}/api/state", timeout=REQUEST_TIMEOUT)
        return jsonify(r.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@app.route("/speed", methods=["POST"])
def speed():
    payload = request.get_json(force=True)
    left = int(payload.get("left", 0))
    right = int(payload.get("right", 0))
    left = max(-500, min(500, left))
    right = max(-500, min(500, right))
    try:
        r = requests.post(
            f"{ESP32_BASE_URL}/api/speed",
            json={"left": left, "right": right},
            timeout=REQUEST_TIMEOUT,
        )
        return jsonify(r.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@app.route("/stop", methods=["POST"])
def stop():
    try:
        r = requests.post(f"{ESP32_BASE_URL}/api/stop", timeout=REQUEST_TIMEOUT)
        return jsonify(r.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@app.route("/clear_bump", methods=["POST"])
def clear_bump():
    try:
        r = requests.post(f"{ESP32_BASE_URL}/api/clear_bump", timeout=REQUEST_TIMEOUT)
        return jsonify(r.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502

@app.route("/power_down", methods=["POST"])
def power_down():
    try:
        # El ESP32 entra en deep sleep tras responder, asi que un timeout
        # o una conexion cortada aqui son el comportamiento esperado, no un fallo.
        r = requests.post(f"{ESP32_BASE_URL}/api/power_down", timeout=REQUEST_TIMEOUT)
        return jsonify(r.json())
    except requests.RequestException as e:
        return jsonify({"ok": True, "info": "El ESP32 pudo haberse apagado antes de responder", "detail": str(e)})

@app.route("/calibrate", methods=["POST"])
def calibrate():
    try:
        # La calibracion mueve el robot ~3.5s, asi que el timeout debe ser mayor al general
        r = requests.post(f"{ESP32_BASE_URL}/api/calibrate", timeout=6)
        return jsonify(r.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
