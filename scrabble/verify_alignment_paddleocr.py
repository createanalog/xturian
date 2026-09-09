# -*- coding: utf-8 -*-
"""
Depuración visual de PaddleOCR: dibuja directamente los cuadros
delimitadores (dt_polys) y el texto reconocido (rec_texts) que devuelve
PaddleOCR sobre el tablero, SIN pasar por la lógica de asignación a
casillas de recognize_board() / _distribuir_texto_en_celdas().

Útil para ver exactamente qué está detectando PaddleOCR - cómo quedan
agrupadas las palabras, qué tan ajustados son los cuadros a las fichas
reales, si el eje del cuadro es horizontal o vertical - antes de decidir
cómo repartir esos cuadros entre casillas.

El OCR corre en un hilo separado del principal (igual que en
verify_alignment.py), así que la vista previa no se congela mientras
PaddleOCR procesa.

No requiere haber calibrado el tablero vacío (a diferencia de
verify_alignment.py) - este script no filtra por ocupación, solo muestra
crudo lo que PaddleOCR devuelve.

Uso:
    python verify_alignment_paddleocr.py --camera 0

Controles:
    [q] o [ESC]  -> salir
"""

import argparse
import threading
import time

import cv2

import scrabble_processor as sp
import vision_manager as vm
import lrecog_paddleocr as paddle_engine
from board_config import BOARD_SIZE

WINDOW_NAME = "PaddleOCR - cuadros detectados (q=salir)"
BOX_COLOR = (0, 255, 0)      # verde brillante
TEXT_COLOR = (0, 255, 255)   # amarillo
GRID_COLOR = (90, 90, 90)    # gris tenue, solo de referencia visual


class PaddleBoxWorker:
    """
    Corre PaddleOCR en un hilo separado del principal (mismo patrón que
    OcrWorker en verify_alignment.py), pero guarda los resultados CRUDOS
    (texto, cuadro, confianza) tal cual los devuelve PaddleOCR, en vez de
    una rejilla ya asignada a casillas.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._results = []  # lista de (texto, bbox, score)
        self._busy = False

    def maybe_start(self, image_color):
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        thread = threading.Thread(target=self._run, args=(image_color,), daemon=True)
        thread.start()
        return True

    def _run(self, image_color):
        try:
            reader = paddle_engine.get_ocr_reader()
            raw_results = reader.predict(image_color)
            results = []
            for res in raw_results:
                textos = res.get("rec_texts", [])
                cajas = res.get("dt_polys", [])
                puntajes = res.get("rec_scores", [1.0] * len(textos))
                for texto, bbox, score in zip(textos, cajas, puntajes):
                    results.append((texto, bbox, score))
        except Exception as e:
            print(f"[verify_alignment_paddleocr] Error en el hilo de OCR, se ignora este intento: {e}")
            results = None

        with self._lock:
            if results is not None:
                self._results = results
            self._busy = False

    def get_results(self):
        with self._lock:
            return list(self._results)


def draw_boxes(warped_color, results):
    """
    Dibuja cada cuadro delimitador (dt_polys) y su texto reconocido, tal
    cual los devuelve PaddleOCR - sin ninguna asignación a casillas.
    La rejilla 15x15 se dibuja solo como referencia visual de fondo, en
    gris tenue, para poder comparar a ojo el tamaño de los cuadros contra
    el de una casilla.
    """
    overlay = warped_color.copy()

    for i in range(BOARD_SIZE + 1):
        pos = i * vm.CELL_SIZE
        cv2.line(overlay, (pos, 0), (pos, vm.WARPED_SIZE), GRID_COLOR, 1)
        cv2.line(overlay, (0, pos), (vm.WARPED_SIZE, pos), GRID_COLOR, 1)

    for texto, bbox, score in results:
        pts = bbox.astype(int).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=BOX_COLOR, thickness=2)

        x_min = int(bbox[:, 0].min())
        y_min = int(bbox[:, 1].min())
        etiqueta = f"{texto} ({score:.2f})"
        cv2.putText(
            overlay, etiqueta, (x_min, max(12, y_min - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA,
        )

    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Índice de la webcam (0, 1, ...)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--ocr-interval", type=float, default=2.5,
                         help="Segundos entre relecturas de OCR (corren en un hilo aparte, "
                              "la vista previa nunca se congela esperándolas).")
    args = parser.parse_args()

    print("Cargando modelo de PaddleOCR (puede tardar la primera vez)...")
    paddle_engine.get_ocr_reader()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara #{args.camera}")
        return

    worker = PaddleBoxWorker()
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
            # frame ya es BGR (a color) - PaddleOCR recibe color real,
            # igual que en el pipeline principal.
            warped_color = vm.warp_board(frame, corners)

            now = time.time()
            if (now - last_ocr_trigger) >= args.ocr_interval:
                last_ocr_trigger = now
                worker.maybe_start(warped_color)

            overlay = draw_boxes(warped_color, worker.get_results())
            
            cv2.putText(
                overlay, "usando ultima calibracion conocida",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1,
            )
            cv2.imshow(WINDOW_NAME, overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q o ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
