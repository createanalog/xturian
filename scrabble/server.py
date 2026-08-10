# -*- coding: utf-8 -*-
"""
Servidor que recibe las fotos de la ESP32-CAM, detecta cada jugada de
Scrabble y mantiene el marcador acumulado por jugador.

Uso:
    python server.py --players 2
    (o --players 4, etc. Los jugadores se numeran 1..N y juegan por turnos
     en ese orden, ronda tras ronda - ver README.md para cómo cambiarlo)

Endpoints:
    POST /upload      <- la ESP32-CAM sube aquí cada foto (body = JPEG binario)
    GET  /score        -> JSON con el marcador y el historial de jugadas
    GET  /              -> panel web simple para ver el marcador en vivo
"""

import argparse
import io
import json
import os
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

import scrabble_processor as sp
from board_config import BOARD_SIZE

app = Flask(__name__)

STATE_FILE = "game_state.json"

# Cuántas fotos "iguales" seguidas exigimos antes de considerar el tablero
# estable (evita procesar mientras una mano todavía se está moviendo).
STABILITY_FRAMES = 3
STABILITY_DIFF_THRESHOLD = 8  # diferencia media de píxel entre frames consecutivos


class GameState:
    def __init__(self, num_players):
        self.num_players = num_players
        self.current_player = 1
        self.scores = {i: 0 for i in range(1, num_players + 1)}
        self.history = []  # lista de dicts: jugador, palabras, puntos, timestamp
        self.grid = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.recent_frames = deque(maxlen=STABILITY_FRAMES)
        self.empty_cells = None
        self.templates = None

    def load_calibration(self):
        self.empty_cells = sp.load_empty_board_cells()
        self.templates = sp.load_letter_templates()

    def save(self):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "scores": self.scores,
                "history": self.history,
                "current_player": self.current_player,
            }, f, ensure_ascii=False, indent=2)

    def to_dict(self):
        return {
            "scores": self.scores,
            "current_player": self.current_player,
            "history": self.history,
        }


game = None  # se inicializa en main()


def frames_are_stable(frames):
    if len(frames) < STABILITY_FRAMES:
        return False
    ref = frames[0]
    for f in list(frames)[1:]:
        if f.shape != ref.shape:
            return False
        diff = float(np.mean(cv2.absdiff(f, ref)))
        if diff > STABILITY_DIFF_THRESHOLD:
            return False
    return True


@app.route("/upload", methods=["POST"])
def upload():
    raw = request.get_data()
    if not raw:
        return jsonify({"status": "error", "msg": "sin datos"}), 400

    npimg = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"status": "error", "msg": "JPEG inválido"}), 400

    small_gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (200, 200))
    game.recent_frames.append(small_gray)

    if not frames_are_stable(game.recent_frames):
        return jsonify({"status": "esperando_estabilidad"}), 200

    grid, ok = sp.read_board(image, game.empty_cells, game.templates)
    if not ok:
        return jsonify({"status": "tablero_no_detectado"}), 200

    new_positions = sp.diff_new_tiles(game.grid, grid)

    # Si hay letras "desconocidas" (None) entre las nuevas casillas ocupadas,
    # no arriesgamos a puntuar mal: se espera a una foto más nítida.
    if not new_positions:
        return jsonify({"status": "sin_cambios"}), 200

    unresolved = [pos for pos in new_positions if grid[pos[0]][pos[1]] is None]
    if unresolved:
        return jsonify({"status": "letras_no_reconocidas", "casillas": unresolved}), 200

    # Validar que las fichas nuevas están alineadas (misma fila o misma columna)
    rows = {r for r, c in new_positions}
    cols = {c for r, c in new_positions}
    if len(rows) > 1 and len(cols) > 1:
        return jsonify({"status": "jugada_no_alineada", "casillas": new_positions}), 200

    points, detail = sp.score_play(grid, new_positions)

    game.grid = grid
    game.scores[game.current_player] += points
    game.history.append({
        "jugador": game.current_player,
        "casillas": new_positions,
        "palabras": detail,
        "puntos": points,
        "acumulado": game.scores[game.current_player],
        "timestamp": time.time(),
    })
    game.current_player = (game.current_player % game.num_players) + 1
    game.save()

    return jsonify({
        "status": "jugada_registrada",
        "puntos": points,
        "detalle": detail,
        "marcador": game.scores,
        "siguiente_turno": game.current_player,
    }), 200


@app.route("/next_turn", methods=["POST", "GET"])
def next_turn():
    """
    Cambio de turno manual (botón físico en la ESP32-CAM, o llamado a mano).
    Útil cuando un jugador pasa turno, cambia fichas, o hay que corregir
    la rotación automática por algún motivo.
    """
    jugador_anterior = game.current_player
    game.current_player = (game.current_player % game.num_players) + 1
    game.history.append({
        "jugador": jugador_anterior,
        "casillas": [],
        "palabras": [],
        "puntos": 0,
        "acumulado": game.scores[jugador_anterior],
        "manual": True,
        "nota": "Cambio de turno manual (botón)",
        "timestamp": time.time(),
    })
    game.save()
    return jsonify({
        "status": "turno_cambiado",
        "jugador_anterior": jugador_anterior,
        "turno_actual": game.current_player,
    }), 200


@app.route("/score", methods=["GET"])
def score():
    return jsonify(game.to_dict())


PANEL_HTML = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Marcador Scrabble</title>
<meta http-equiv="refresh" content="4">
<style>
  body { font-family: sans-serif; background:#222; color:#eee; padding:2rem; }
  h1 { color:#f5c518; }
  .score { font-size:2rem; margin:0.3rem 0; }
  table { border-collapse: collapse; margin-top:1rem; width:100%; }
  td, th { border:1px solid #555; padding:6px 10px; text-align:left; }
  .turno { color:#66ff66; font-weight:bold; }
</style>
</head>
<body>
<h1>Marcador Scrabble</h1>
{% for jugador, puntos in scores.items() %}
  <div class="score">Jugador {{ jugador }}: {{ puntos }} puntos
    {% if jugador == current_player %}<span class="turno"> (turno actual)</span>{% endif %}
  </div>
{% endfor %}

<h2>Historial de jugadas</h2>
<table>
<tr><th>#</th><th>Jugador</th><th>Palabras</th><th>Puntos</th><th>Acumulado</th></tr>
{% for jugada in history|reverse %}
  <tr>
    <td>{{ loop.revindex }}</td>
    <td>{{ jugada.jugador }}</td>
    <td>{% if jugada.manual %}(cambio de turno manual){% else %}{{ jugada.palabras|map(attribute=0)|join(', ') }}{% endif %}</td>
    <td>{{ jugada.puntos }}</td>
    <td>{{ jugada.acumulado }}</td>
  </tr>
{% endfor %}
</table>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def panel():
    d = game.to_dict()
    return render_template_string(
        PANEL_HTML,
        scores=d["scores"],
        current_player=d["current_player"],
        history=d["history"],
    )


def main():
    global game
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=2, help="Número de jugadores")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    game = GameState(args.players)
    print("Cargando calibración (tablero vacío + plantillas de letras)...")
    game.load_calibration()
    print("Calibración cargada. Servidor listo.")
    print(f"Panel web: http://0.0.0.0:{args.port}/")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
