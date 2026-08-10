# -*- coding: utf-8 -*-
"""
Núcleo de visión por computador y puntuación para el Scrabble.

Flujo general por cada foto recibida:
  1. Localizar el tablero mediante 4 marcadores ArUco pegados en las esquinas.
  2. Corregir la perspectiva y recortar la rejilla 15x15.
  3. Determinar qué casillas están ocupadas (comparando contra el tablero
     vacío de referencia).
  4. Reconocer la letra de cada casilla ocupada por template matching contra
     la biblioteca de plantillas generada en la calibración.
  5. Comparar contra el estado anterior -> nuevas fichas colocadas.
  6. Si las nuevas fichas forman una jugada válida (alineadas en una fila o
     columna, contiguas), calcular su puntuación oficial.

IMPORTANTE - calibración necesaria antes de jugar (ver README.md):
  - board_templates/empty_board.jpg   -> foto del tablero vacío
  - board_templates/letters/<LETRA>.png -> una plantilla recortada por letra
"""

import os
import json
import glob
import cv2
import numpy as np

from board_config import BOARD_SIZE, BOARD_LAYOUT, LETTER_VALUES, BINGO_BONUS, BINGO_TILE_COUNT

TEMPLATES_DIR = "board_templates"
LETTERS_DIR = os.path.join(TEMPLATES_DIR, "letters")
EMPTY_BOARD_PATH = os.path.join(TEMPLATES_DIR, "empty_board.jpg")

WARPED_SIZE = 600           # tablero corregido -> 600x600 px
CELL_SIZE = WARPED_SIZE // BOARD_SIZE
OCCUPIED_DIFF_THRESHOLD = 25   # sensibilidad para detectar "hay ficha aquí"
MATCH_MIN_SCORE = 0.35         # umbral mínimo de confianza del template matching

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
    """Devuelve una matriz BOARD_SIZE x BOARD_SIZE de recortes (imágenes) de cada casilla."""
    cells = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            y0, y1 = r * CELL_SIZE, (r + 1) * CELL_SIZE
            x0, x1 = c * CELL_SIZE, (c + 1) * CELL_SIZE
            # margen interior para evitar bordes de la rejilla
            margin = int(CELL_SIZE * 0.12)
            cell = warped_gray[y0 + margin:y1 - margin, x0 + margin:x1 - margin]
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


def load_letter_templates():
    """Carga plantillas <LETRA>.png de board_templates/letters/ y las redimensiona
    al tamaño de celda para comparación directa."""
    templates = {}
    if not os.path.isdir(LETTERS_DIR):
        raise FileNotFoundError(
            f"Falta la carpeta {LETTERS_DIR} con las plantillas de letras. Ver README.md."
        )
    for path in glob.glob(os.path.join(LETTERS_DIR, "*.png")):
        letter = os.path.splitext(os.path.basename(path))[0].upper()
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cell_px = CELL_SIZE - 2 * int(CELL_SIZE * 0.12)
        img = cv2.resize(img, (cell_px, cell_px))
        templates[letter] = img
    if not templates:
        raise RuntimeError("No se cargó ninguna plantilla de letra. Revisa board_templates/letters/.")
    return templates


# ---------------------------------------------------------------------------
# Reconocimiento de una celda: vacía / ocupada / qué letra
# ---------------------------------------------------------------------------
def cell_is_occupied(cell_gray, empty_ref_gray):
    diff = cv2.absdiff(cell_gray, empty_ref_gray)
    return float(np.mean(diff)) > OCCUPIED_DIFF_THRESHOLD


def recognize_letter(cell_gray, templates):
    """Compara la celda contra todas las plantillas y devuelve (letra, score)."""
    best_letter, best_score = None, -1.0
    cell_norm = cv2.equalizeHist(cell_gray)
    for letter, template in templates.items():
        result = cv2.matchTemplate(cell_norm, cv2.equalizeHist(template), cv2.TM_CCOEFF_NORMED)
        score = float(result.max())
        if score > best_score:
            best_score = score
            best_letter = letter
    if best_score < MATCH_MIN_SCORE:
        return None, best_score
    return best_letter, best_score


def read_board(image, empty_cells, templates):
    """
    Procesa una foto completa del tablero y devuelve una matriz BOARD_SIZE x BOARD_SIZE
    con la letra en cada casilla ocupada o None si está vacía.
    Devuelve (grid, warped_ok:bool)
    """
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners = find_board_corners(image)
    if corners is None:
        return None, False

    warped = warp_board(gray_full, corners)
    cells = extract_cells(warped)

    grid = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if cell_is_occupied(cells[r][c], empty_cells[r][c]):
                letter, score = recognize_letter(cells[r][c], templates)
                grid[r][c] = letter  # puede ser None si no hay match seguro
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
