# -*- coding: utf-8 -*-
"""
Núcleo de visión por computador y puntuación para el Scrabble.

Flujo general por cada foto recibida:
  1. Localizar el tablero mediante 4 marcadores ArUco pegados en las esquinas.
  2. Corregir la perspectiva y recortar la rejilla 15x15.
  3. Determinar qué casillas están ocupadas (comparando contra el tablero
     vacío de referencia).
  4. Reconocer la letra de cada casilla ocupada con EasyOCR (red neuronal,
     tolerante a ruido/blur/iluminación variable).
  5. Comparar contra el estado anterior -> nuevas fichas colocadas.
  6. Si las nuevas fichas forman una jugada válida (alineadas en una fila o
     columna, contiguas), calcular su puntuación oficial.

IMPORTANTE - calibración necesaria antes de jugar (ver README.md):
  - board_templates/empty_board.jpg   -> foto del tablero vacío
  (ya NO hace falta calibrar plantillas por letra: EasyOCR no las necesita)
"""

import os
import importlib
import cv2
import numpy as np

from board_config import BOARD_SIZE, BOARD_LAYOUT, LETTER_VALUES, BINGO_BONUS, BINGO_TILE_COUNT

# ---------------------------------------------------------------------------
# Selección del motor de OCR (easyocr / ocrad / tesseract)
# ---------------------------------------------------------------------------
# Cada motor vive en su propio módulo (lrecog_<nombre>.py) con la misma
# interfaz: get_ocr_reader() y recognize_letter(cell_gray). Se importa de
# forma PEREZOSA (recién cuando se selecciona o se usa por primera vez)
# para no forzar tener instaladas las librerías/binarios de los 3 motores
# si solo vas a usar uno.
_OCR_ENGINE_MODULES = {
    "easyocr": "lrecog_easyocr",
    "ocrad": "lrecog_ocrad",
    "tesseract": "lrecog_tesseract",
}
DEFAULT_OCR_ENGINE = "tesseract"

_active_engine = None
_active_engine_name = DEFAULT_OCR_ENGINE


def set_ocr_engine(name):
    """Selecciona qué motor de OCR usar: 'easyocr', 'ocrad' o 'tesseract'."""
    global _active_engine, _active_engine_name
    name = name.lower()
    if name not in _OCR_ENGINE_MODULES:
        raise ValueError(
            f"Motor de OCR desconocido: {name!r}. Opciones válidas: "
            f"{list(_OCR_ENGINE_MODULES)}"
        )
    _active_engine = importlib.import_module(_OCR_ENGINE_MODULES[name])
    _active_engine_name = name


def _engine():
    global _active_engine
    if _active_engine is None:
        # Nadie llamó a set_ocr_engine() todavía -> se importa el motor por
        # defecto recién ahora que hace falta de verdad.
        set_ocr_engine(_active_engine_name)
    return _active_engine


def get_ocr_reader():
    return _engine().get_ocr_reader()


def recognize_letter(cell_gray):
    return _engine().recognize_letter(cell_gray)

TEMPLATES_DIR = "board_templates"
EMPTY_BOARD_PATH = os.path.join(TEMPLATES_DIR, "empty_board.jpg")

WARPED_SIZE = 600           # tablero corregido -> 600x600 px
CELL_SIZE = WARPED_SIZE // BOARD_SIZE
OCCUPIED_DIFF_THRESHOLD = 25   # sensibilidad para detectar "hay ficha aquí"

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()


# ---------------------------------------------------------------------------
# Detección del tablero mediante marcadores ArUco en las 4 esquinas
# ---------------------------------------------------------------------------
def find_board_corners(image):
    """
    Espera 4 marcadores ArUco (ids 0,1,2,3) pegados justo fuera de las
    esquinas del tablero, en este orden: 0=sup-izq, 1=sup-der, 2=inf-der, 3=inf-izq.
    Devuelve los 4 puntos (centros de los marcadores) o None si no se detectan los 4.
    """
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or len(ids) < 4:
        return None

    ids = ids.flatten()
    needed = {0, 1, 2, 3}
    if not needed.issubset(set(ids)):
        return None

    centers = {}
    for i, marker_id in enumerate(ids):
        if marker_id in needed:
            pts = corners[i][0]
            centers[marker_id] = pts.mean(axis=0)

    ordered = np.array([centers[0], centers[1], centers[2], centers[3]], dtype=np.float32)
    return ordered


# Recuerda la última detección válida de los 4 marcadores. Se reutiliza
# cuando un frame puntual falla en detectarlos (por ejemplo, una mano
# tapando momentáneamente un marcador, o un frame con motion blur), en vez
# de descartar ese frame por completo.
#
# IMPORTANTE: esto asume que, mientras los marcadores no se detectan de
# nuevo, el tablero no se movió. Si el tablero se mueve justo durante ese
# lapso sin detección, la calibración cacheada queda desactualizada hasta
# la siguiente detección real.
_last_known_corners = None


def find_board_corners_cached(image):
    """
    Igual que find_board_corners(), pero si en este frame no se detectan
    los 4 marcadores, reutiliza la última detección válida en vez de
    devolver None.

    Devuelve (corners, detected_now):
      - corners: los 4 puntos a usar (de este frame o cacheados), o None si
        todavía no hubo NINGUNA detección válida desde que arrancó el proceso.
      - detected_now: True si los marcadores se detectaron en este frame
        (calibración "fresca"), False si se está reutilizando la caché.
    """
    global _last_known_corners
    corners = find_board_corners(image)
    if corners is not None:
        _last_known_corners = corners
        return corners, True
    return _last_known_corners, False


def reset_corner_cache():
    """Olvida la última detección cacheada (por ejemplo, si sabes que
    moviste el tablero o la cámara y no quieres arrastrar una calibración
    vieja hasta la próxima detección real)."""
    global _last_known_corners
    _last_known_corners = None


def warp_board(image, src_points):
    dst_points = np.array([
        [0, 0],
        [WARPED_SIZE - 1, 0],
        [WARPED_SIZE - 1, WARPED_SIZE - 1],
        [0, WARPED_SIZE - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (WARPED_SIZE, WARPED_SIZE))
    return warped


def extract_cells(warped_gray):
    """Devuelve una matriz BOARD_SIZE x BOARD_SIZE de recortes (imágenes) de cada casilla.
    Se usa la celda completa, sin recortar margen interior."""
    cells = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            y0, y1 = r * CELL_SIZE, (r + 1) * CELL_SIZE
            x0, x1 = c * CELL_SIZE, (c + 1) * CELL_SIZE
            cell = warped_gray[y0:y1, x0:x1]
            cells[r][c] = cell
    return cells


# ---------------------------------------------------------------------------
# Calibración: tablero vacío y plantillas de letras
# ---------------------------------------------------------------------------
def load_empty_board_cells():
    if not os.path.exists(EMPTY_BOARD_PATH):
        raise FileNotFoundError(
            f"Falta {EMPTY_BOARD_PATH}. Ejecuta la calibración primero (ver README.md)."
        )
    img = cv2.imread(EMPTY_BOARD_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = find_board_corners(img)
    if corners is None:
        raise RuntimeError("No se detectaron los 4 marcadores ArUco en la foto del tablero vacío.")
    warped = warp_board(gray, corners)
    return extract_cells(warped)


# ---------------------------------------------------------------------------
# Reconocimiento de una celda: vacía / ocupada / qué letra
# ---------------------------------------------------------------------------
def cell_is_occupied(cell_gray, empty_ref_gray):
    diff = cv2.absdiff(cell_gray, empty_ref_gray)
    return float(np.mean(diff)) > OCCUPIED_DIFF_THRESHOLD


def read_board(image, empty_cells):
    """
    Procesa una foto completa del tablero y devuelve una matriz BOARD_SIZE x BOARD_SIZE
    con la letra en cada casilla ocupada o None si está vacía.
    Devuelve (grid, warped_ok:bool)

    Si los marcadores no se detectan en este frame puntual, se reutiliza la
    última calibración válida conocida (ver find_board_corners_cached) en
    vez de descartar el frame.
    """
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, detected_now = find_board_corners_cached(image)
    if corners is None:
        return None, False

    warped = warp_board(gray_full, corners)
    cells = extract_cells(warped)

    grid = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if cell_is_occupied(cells[r][c], empty_cells[r][c]):
                letter, confidence = recognize_letter(cells[r][c])
                grid[r][c] = letter  # puede ser None si no hay lectura confiable
    return grid, True


# ---------------------------------------------------------------------------
# Lógica de jugada y puntuación (reglas oficiales de Scrabble)
# ---------------------------------------------------------------------------
def diff_new_tiles(prev_grid, new_grid):
    """Casillas que estaban vacías y ahora tienen letra."""
    new_positions = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            was_empty = prev_grid[r][c] is None
            now_filled = new_grid[r][c] is not None
            if was_empty and now_filled:
                new_positions.append((r, c))
    return new_positions


def _word_positions_through(grid, r, c, horizontal):
    """Devuelve la lista de posiciones (r,c) que forman la palabra continua
    que pasa por (r,c) en la dirección indicada."""
    positions = [(r, c)]
    if horizontal:
        cc = c - 1
        while cc >= 0 and grid[r][cc] is not None:
            positions.insert(0, (r, cc))
            cc -= 1
        cc = c + 1
        while cc < BOARD_SIZE and grid[r][cc] is not None:
            positions.append((r, cc))
            cc += 1
    else:
        rr = r - 1
        while rr >= 0 and grid[rr][c] is not None:
            positions.insert(0, (rr, c))
            rr -= 1
        rr = r + 1
        while rr < BOARD_SIZE and grid[rr][c] is not None:
            positions.append((rr, c))
            rr += 1
    return positions


def score_play(grid, new_positions):
    """
    Calcula la puntuación total de una jugada.
    grid: estado completo del tablero DESPUÉS de la jugada (letra o None por celda)
    new_positions: lista de (r,c) recién colocadas en este turno
    Devuelve (puntuacion_total, detalle_palabras: list[(palabra, puntos)])
    """
    if not new_positions:
        return 0, []

    new_set = set(new_positions)
    words_seen = set()   # para no puntuar la misma palabra 2 veces
    detail = []
    total = 0

    # Palabra principal: todas las nuevas fichas están en la misma fila o columna
    rows = {r for r, c in new_positions}
    cols = {c for r, c in new_positions}
    directions_to_check = []

    if len(rows) == 1:
        directions_to_check.append((next(iter(rows)), next(iter(new_positions))[1], True))
    if len(cols) == 1:
        directions_to_check.append((next(iter(new_positions))[0], next(iter(cols)), False))

    # Palabra principal (si existe una dirección clara)
    for (r, c, horizontal) in directions_to_check:
        positions = _word_positions_through(grid, r, c, horizontal)
        if len(positions) > 1:
            key = ("H" if horizontal else "V", positions[0], positions[-1])
            if key not in words_seen:
                words_seen.add(key)
                word_str, pts = _score_word(grid, positions, new_set)
                detail.append((word_str, pts))
                total += pts

    # Palabras cruzadas formadas por cada ficha nueva individual
    for (r, c) in new_positions:
        for horizontal in (True, False):
            positions = _word_positions_through(grid, r, c, horizontal)
            if len(positions) > 1:
                key = ("H" if horizontal else "V", positions[0], positions[-1])
                if key not in words_seen:
                    words_seen.add(key)
                    word_str, pts = _score_word(grid, positions, new_set)
                    detail.append((word_str, pts))
                    total += pts

    if len(new_positions) == BINGO_TILE_COUNT:
        total += BINGO_BONUS
        detail.append((f"BONUS por usar {BINGO_TILE_COUNT} fichas", BINGO_BONUS))

    return total, detail


def _score_word(grid, positions, new_set):
    word_str = ""
    letter_sum = 0
    word_multiplier = 1

    for (r, c) in positions:
        letter = grid[r][c]
        word_str += letter if letter else "?"
        base_value = LETTER_VALUES.get(letter, 0)
        letter_value = base_value

        # Las casillas premium sólo cuentan si la ficha se colocó en ESTE turno
        if (r, c) in new_set:
            premium = BOARD_LAYOUT[r][c]
            if premium == "DL":
                letter_value *= 2
            elif premium == "TL":
                letter_value *= 3
            elif premium == "DW":
                word_multiplier *= 2
            elif premium == "TW":
                word_multiplier *= 3

        letter_sum += letter_value

    return word_str, letter_sum * word_multiplier
