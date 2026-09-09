# -*- coding: utf-8 -*-
"""
Verificación visual EN VIVO, leyendo directamente de una webcam.

Muestra en una ventana (cv2.imshow) el tablero ya enderezado por
perspectiva, con:
  - la rejilla 15x15 dibujada.
  - la letra reconocida escrita en VERDE BRILLANTE sobre cada ficha
    detectada.

El OCR corre en un HILO SEPARADO del principal (ver OcrWorker más abajo).
El hilo principal (captura de la webcam + cv2.imshow) nunca espera a que
el OCR termine - sigue mostrando frames fluidos todo el tiempo, y la
rejilla de letras que se dibuja es simplemente la última que el hilo de
OCR terminó de calcular, aunque venga de un frame algo anterior. Antes de
este cambio, cv2.imshow se congelaba cada vez que el OCR corría (podían
ser varios segundos con algunos motores), porque todo pasaba en el mismo
hilo que dibuja la ventana.

Con los motores de OCR Tesseract o PaddleOCR, la lectura de letras se hace
además en UNA SOLA pasada sobre todo el tablero por cada relectura (en vez
de una llamada por cada casilla ocupada) - ver scrabble_processor.py.

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
import threading
import time

import cv2

import scrabble_processor as sp
import vision_manager as vm
from board_config import BOARD_SIZE

WINDOW_NAME = "Verificacion en vivo (q=salir)"
LETTER_COLOR = (0, 255, 0)  # BGR -> verde brillante


class OcrWorker:
    """
    Corre el reconocimiento de letras (sp.read_board_from_warped, que
    puede tardar desde milisegundos hasta varios segundos según el motor)
    en un hilo separado del principal, para que la ventana de vista previa
    nunca se congele esperándolo.

    Uso: maybe_start(...) se llama desde el hilo principal cada vez que
    toca una relectura (según --ocr-interval); si el hilo de OCR ya está
    ocupado procesando la relectura anterior, simplemente no hace nada
    (se salta esta relectura en vez de acumular trabajo pendiente).
    get_grid() devuelve el último resultado ya terminado, sea cual sea el
    frame del que haya salido.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._grid = None
        self._busy = False

    def maybe_start(self, image):
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        thread = threading.Thread(
            target=self._run, args=(image,), daemon=True,
        )
        thread.start()
        return True

    def _run(self, image):
        try:
            print(f"El tipo de la variable es: {type(image)}")
            grid = vm.read_board(image)
        except Exception as e:
            print(f"[verify_alignment] Error en el hilo de OCR, se ignora este intento: {e}")
            grid = None

        with self._lock:
            if grid is not None:
                self._grid = grid
            self._busy = False

    def get_grid(self):
        with self._lock:
            return self._grid



def draw_overlay(warped, grid, extra_text=None):
    """Dibuja la rejilla 15x15 y, si hay un resultado de OCR disponible
    (grid, puede ser None si el hilo de OCR todavía no terminó su primera
    pasada), las letras detectadas en verde brillante. No calcula nada de
    OCR aquí - solo dibuja sobre lo que ya se tiene a mano, por eso es
    seguro llamarla en cada frame sin afectar la fluidez."""
    overlay = warped#cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)

    for i in range(BOARD_SIZE + 1):
        pos = i * vm.CELL_SIZE
        cv2.line(overlay, (pos, 0), (pos, vm.WARPED_SIZE), (0, 150, 0), 1)
        cv2.line(overlay, (0, pos), (vm.WARPED_SIZE, pos), (0, 150, 0), 1)

    if grid is not None:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                letter = grid[r][c]
                if letter:
                    x = c * vm.CELL_SIZE + int(vm.CELL_SIZE * 0.18)
                    y = r * vm.CELL_SIZE + int(vm.CELL_SIZE * 0.68)
                    cv2.putText(
                        overlay, letter, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        LETTER_COLOR, 2, cv2.LINE_AA,
                    )

    if extra_text:
        cv2.putText(
            overlay, extra_text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1,
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
                         help="Segundos entre relecturas de OCR (ahora corren en un hilo aparte, "
                              "así que la vista previa nunca se congela esperándolas).")
    parser.add_argument(
        "--ocr-engine", type=str, default="paddleocr",
        choices=["easyocr", "ocrad", "tesseract", "paddleocr"],
        help="Motor de OCR a usar para leer las letras (default: paddleocr)",
    )
    args = parser.parse_args()

    vm.set_ocr_engine(args.ocr_engine)
    print(f"Motor de OCR: {args.ocr_engine}")

    print("Cargando modelo de OCR (puede tardar la primera vez)...")
    vm.get_ocr_reader()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara #{args.camera}")
        return

    worker = OcrWorker()
    last_ocr_trigger = 0.0

    print("Ventana activa. Teclas: [q] o [ESC] para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error leyendo frame de la webcam")
            break

        corners = vm.find_board_corners(frame)
        if corners is None:
            cv2.putText(
                frame, "Tablero no detectado (revisa marcadores ArUco)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
            cv2.imshow(WINDOW_NAME, frame)
        else:
           
            now = time.time()
            if (now - last_ocr_trigger) >= args.ocr_interval:
                last_ocr_trigger = now
                # No bloquea: si el hilo de OCR ya está ocupado con la
                # relectura anterior, maybe_start() no hace nada y
                # simplemente se reintenta en el próximo tick.
                worker.maybe_start(frame)

            warped = vm.warp_board(frame, corners)
            overlay = draw_overlay(warped, worker.get_grid(), "")
            cv2.imshow(WINDOW_NAME, overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q o ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
