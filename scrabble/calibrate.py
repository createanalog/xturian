# -*- coding: utf-8 -*-
"""
Ayudante de calibración.

Único paso necesario - Tablero vacío:
    python calibrate.py empty foto_tablero_vacio.jpg
    -> guarda board_templates/empty_board.jpg, usado para detectar qué
       casillas están ocupadas comparando contra este tablero de referencia.

Ya NO hace falta calibrar plantillas por letra: el reconocimiento de letras
usa EasyOCR (ver scrabble_processor.py), que no requiere ejemplos previos
de cada letra.
"""

import sys
import os
import cv2

import scrabble_processor as sp

TEMPLATES_DIR = "board_templates"


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


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "empty":
        print(__doc__)
        return
    cmd_empty(sys.argv[2])


if __name__ == "__main__":
    main()
