# -*- coding: utf-8 -*-
"""
Scrabble game counter server

Use:
    python server.py --players 2
  
Endpoints:
    POST /upload      <- to upload video image
    GET  /score        -> outputs JSON
    GET  /              -> Web panel
"""

import argparse
import json
import os
import threading
import time
import logging

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string, redirect, url_for

import scrabble_processor as sp
import vision_manager as vm
from board_config import BOARD_SIZE

# Configuración GLOBAL (solo aquí)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


app = Flask(__name__)

STATE_FILE = "game_state.json"

last_image = None
state_lock = threading.Lock()


class GameState:
    def __init__(self, num_players):
        self.num_players = num_players
        self.current_player = 1
        self.scores = {i: 0 for i in range(1, num_players + 1)}
        self.history = []  # jugador, palabras, puntos, timestamp
        self.grid = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.empty_cells = None

    def save(self):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "scores": self.scores,
                "current_player": self.current_player,
                 "history": self.history,
            }, f, ensure_ascii=False, indent=2)

    def to_dict(self):
        return {
            "scores": self.scores,
            "current_player": self.current_player,
            "history": self.history,
        }


game = None  # se inicializa en main()


@app.route("/upload", methods=["POST"])
def upload():
    
    raw = request.get_data()
    if not raw:
        return jsonify({"status": "error", "msg": "no data"}), 400

    npimg = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"status": "error", "msg": "Invalid jpeg"}), 400

    global last_image
    
    with state_lock:
        last_image = image
        scores = dict(game.scores)
        siguiente_turno = game.current_player

    vm.imageSeen(last_image)

    return jsonify({
        "status": "image_upload_ok",
        "marcador": scores,
        "siguiente_turno": siguiente_turno,
    }), 200
    
    
@app.route("/play", methods=["POST"])
def play():

    with state_lock:             # TODO: This can take a loooong time.
        
        if last_image is None:
            return jsonify({"status": "No game image"}), 200

        grid = vm.read_board(last_image)

        if grid is None:
            return jsonify({"status": "Can't detect Board"}), 200    # TODO: define a STATUS field in panel screen

        new_positions = sp.find_new_tiles(game.grid, grid)

        if not new_positions:
            return jsonify({"status": "no_changes"}), 200

        rows = {r for r, c in new_positions}
        cols = {c for r, c in new_positions}
        if len(rows) > 1 and len(cols) > 1:
            return jsonify({"status": "non aligned play", "positions": new_positions}), 200

        points, detail = sp.score_play(grid, new_positions)

        game.grid = grid
        game.scores[game.current_player] += points
        game.history.append({
            "jugador": game.current_player,
            "casillas": list(new_positions),
            "palabras": detail,
            "puntos": points,
            "acumulado": game.scores[game.current_player],
            "timestamp": time.time(),
        })
        game.current_player = (game.current_player % game.num_players) + 1
        game.save()

    return redirect(url_for("panel"))
    

@app.route("/score", methods=["GET"])
def score():
    return jsonify(game.to_dict())


@app.route("/", methods=["GET"])
def panel():

    with state_lock:
        d = game.to_dict()

    return render_template_string(
        PANEL_HTML,
        scores=d["scores"],
        current_player=d["current_player"],
        history=d["history"],
    )


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

<form action="/play" method="post" style="margin-bottom: 1rem;">
  <button type="submit" style="padding: 10px 20px; font-size: 1rem; cursor: pointer;">
    📸 Process Play / Turn
  </button>
</form>

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


def main():
    
    global game
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=2, help="Players number")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--ocr-engine", type=str, default="paddleocr",
        choices=["easyocr",  "paddleocr"],
        help="OCR engine: paddleocr (default) or easyocr)",
    )
    args = parser.parse_args()

    vm.set_ocr_engine(args.ocr_engine)
    print(f"OCR engine: {args.ocr_engine}")

    game = GameState(args.players)

    print(f"Web: http://0.0.0.0:{args.port}/")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main()
