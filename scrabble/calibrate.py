# -*- coding: utf-8 -*-
"""
Ayudante de calibración.

Paso 1 - Tablero vacío:
    python calibrate.py empty foto_tablero_vacio.jpg
    -> guarda board_templates/empty_board.jpg (recorte y perspectiva ya
       corregida se hacen en tiempo real al leerla, aquí solo se valida
       que los 4 marcadores ArUco se detectan).

Paso 2 - Plantillas de letras:
    Coloca UNA ficha en una casilla concreta del tablero (por ejemplo, la
    esquina superior izquierda, fila 0 columna 0) y ejecuta:
        python calibrate.py letter foto.jpg A 0 0
    Repite para cada letra (A, B, C, ... Ñ, CH, LL, RR según tu set).
    Esto recorta esa casilla exacta y la guarda como plantilla de "A".
"""

import sys
import os
import cv2

import scrabble_processor as sp
from board_config import BOARD_SIZE

TEMPLATES_DIR = "board_templates"
LETTERS_DIR = os.path.join(TEMPLATES_DIR, "letters")


def cmd_empty(photo_path):
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    image = cv2.imread(photo_path)
    if image is None:
        print(f"No se pudo leer {photo_path}")
        return
    corners = sp.find_board_corners(image)
    if corners is None:
        print("ERROR: no se detectaron los 4 marcadores ArUco (ids 0,1,2,3) en la foto.")
        print("Revisa que estén bien pegados, visibles y con buena luz.")
        return
    dest = os.path.join(TEMPLATES_DIR, "empty_board.jpg")
    cv2.imwrite(dest, image)
    print(f"OK -> guardado {dest}")


def cmd_letter(photo_path, letter, row, col):
    os.makedirs(LETTERS_DIR, exist_ok=True)
    image = cv2.imread(photo_path)
    if image is None:
        print(f"No se pudo leer {photo_path}")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = sp.find_board_corners(image)
    if corners is None:
        print("ERROR: no se detectaron los 4 marcadores ArUco en la foto.")
        return
    warped = sp.warp_board(gray, corners)
    cells = sp.extract_cells(warped)
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        print("Fila/columna fuera de rango (0-14).")
        return
    cell_img = cells[row][col]
    dest = os.path.join(LETTERS_DIR, f"{letter.upper()}.png")
    cv2.imwrite(dest, cell_img)
    print(f"OK -> guardada plantilla de '{letter.upper()}' en {dest}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return

    mode = sys.argv[1]
    if mode == "empty":
        cmd_empty(sys.argv[2])
    elif mode == "letter":
        if len(sys.argv) != 6:
            print("Uso: python calibrate.py letter foto.jpg LETRA FILA COLUMNA")
            return
        _, _, photo_path, letter, row, col = sys.argv
        cmd_letter(photo_path, letter, int(row), int(col))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
