# -*- coding: utf-8 -*-
"""
Verificación visual EN VIVO, leyendo directamente de una webcam.

Muestra en una ventana (cv2.imshow) el tablero ya enderezado por
perspectiva, con:
  - la rejilla 15x15 dibujada.
  - la letra reconocida (EasyOCR) escrita en VERDE BRILLANTE sobre cada
    ficha detectada.

Sirve para comprobar de un vistazo, en tiempo real, si los marcadores
ArUco están bien alineados y si el reconocimiento de letras funciona
razonablemente bien ANTES de arrancar una partida real con server.py.

Si en algún frame puntual no se detectan los 4 marcadores (por ejemplo,
una mano tapando uno momentáneamente), se sigue mostrando el tablero
usando la última calibración válida conocida, con un aviso en amarillo
("usando ultima calibracion conocida") en vez de perder la vista.

Requisito previo: haber calibrado el tablero vacío
    python calibrate.py empty foto_tablero_vacio.jpg

Uso:
    python verify_alignment.py --camera 0

Controles:
    [q] o [ESC]  -> salir
"""

import argparse
import time

import cv2

import scrabble_processor as sp
from board_config import BOARD_SIZE

WINDOW_NAME = "Verificacion en vivo (q=salir)"
LETTER_COLOR = (0, 255, 0)  # BGR -> verde brillante


def draw_grid_and_letters(warped_gray, empty_cells):
    """Devuelve una imagen BGR del tablero enderezado con la rejilla y,
    sobre cada ficha detectada, su letra reconocida en verde brillante."""
    overlay = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)

    # Rejilla 15x15
    for i in range(BOARD_SIZE + 1):
        pos = i * sp.CELL_SIZE
        cv2.line(overlay, (pos, 0), (pos, sp.WARPED_SIZE), (0, 150, 0), 1)
        cv2.line(overlay, (0, pos), (sp.WARPED_SIZE, pos), (0, 150, 0), 1)

    cells = sp.extract_cells(warped_gray)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if sp.cell_is_occupied(cells[r][c], empty_cells[r][c]):
                letter, confidence = sp.recognize_letter(cells[r][c])
                if letter:
                    x = c * sp.CELL_SIZE + int(sp.CELL_SIZE * 0.18)
                    y = r * sp.CELL_SIZE + int(sp.CELL_SIZE * 0.68)
                    cv2.putText(
                        overlay, letter, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        LETTER_COLOR, 2, cv2.LINE_AA,
                    )
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Índice de la webcam (0, 1, ...)")
    parser.add_argument("--width", type=int, default=640,
                         help="Resolución de captura más baja = menos carga de CPU decodificando "
                              "cada frame (el OCR igual trabaja sobre cada celda reescalada, "
                              "así que el ahorro real aquí es moderado).")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--ocr-interval", type=float, default=2.5,
                         help="Segundos entre relecturas de OCR (el OCR es costoso; "
                              "la vista previa sigue fluida entre relecturas).")
    parser.add_argument(
        "--ocr-engine", type=str, default="tesseract",
        choices=["easyocr", "ocrad", "tesseract"],
        help="Motor de OCR a usar para leer las letras (default: tesseract)",
    )
    args = parser.parse_args()

    sp.set_ocr_engine(args.ocr_engine)
    print(f"Motor de OCR: {args.ocr_engine}")

    print("Cargando tablero vacío de referencia...")
    empty_cells = sp.load_empty_board_cells()
    print("Cargando modelo de OCR (puede tardar la primera vez)...")
    sp.get_ocr_reader()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara #{args.camera}")
        return

    print("Ventana activa. Teclas: [q] o [ESC] para salir.")
    last_ocr_time = 0.0
    last_overlay = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error leyendo frame de la webcam")
            break

        corners, detected_now = sp.find_board_corners_cached(frame)
        if corners is None:
            cv2.putText(
                frame, "Tablero no detectado (revisa marcadores ArUco)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
            cv2.imshow(WINDOW_NAME, frame)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            warped_gray = sp.warp_board(gray, corners)

            now = time.time()
            if last_overlay is None or (now - last_ocr_time) >= args.ocr_interval:
                last_ocr_time = now
                last_overlay = draw_grid_and_letters(warped_gray, empty_cells)
                if not detected_now:
                    cv2.putText(
                        last_overlay, "usando ultima calibracion conocida",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1,
                    )

            cv2.imshow(WINDOW_NAME, last_overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q o ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
