# -*- coding: utf-8 -*-
"""
Reemplazo de la ESP32-CAM: captura una webcam conectada a ESTE mismo
computador (donde también corre server.py) y le envía las fotos por HTTP,
exactamente con el mismo protocolo que usaba la placa.

No requiere tocar server.py / scrabble_processor.py / board_config.py:
para el servidor no hay diferencia entre "una foto que llegó de la ESP32"
y "una foto que llegó de este script".

Uso:
    pip install requests opencv-python
    python webcam_client.py --camera 0 --server http://localhost:5000

Controles (con la ventana de vista previa activa):
    [n] o [espacio]  -> cambia de turno (llama a /next_turn)
    [q] o [ESC]       -> salir
"""

import argparse
import time

import cv2
import requests

CAPTURE_INTERVAL_S = 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Índice de la webcam (0, 1, ...)")
    parser.add_argument("--server", type=str, default="http://localhost:5000",
                         help="URL base de server.py")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    args = parser.parse_args()

    upload_url = f"{args.server}/upload"
    next_turn_url = f"{args.server}/next_turn"

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara #{args.camera}")
        return

    print("Webcam iniciada. Vista previa en la ventana; teclas: [n]=turno [q]=salir")
    last_capture = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error leyendo frame de la webcam")
            break

        cv2.imshow("Scrabble - vista previa (n=turno, q=salir)", frame)

        now = time.time()
        if now - last_capture >= CAPTURE_INTERVAL_S:
            last_capture = now
            _, jpg = cv2.imencode(".jpg", frame)
            try:
                resp = requests.post(
                    upload_url,
                    data=jpg.tobytes(),
                    headers={"Content-Type": "image/jpeg"},
                    # El servidor hace todo el análisis (detección de
                    # marcadores + OCR de las 225 casillas) de forma
                    # SÍNCRONA dentro de este mismo request - con motores
                    # como PaddleOCR o Tesseract, sobre todo en la primera
                    # llamada (carga del modelo) o con pocos hilos de CPU,
                    # puede tardar bastante más que unos pocos segundos.
                    timeout=30,
                )
                print(f"[{time.strftime('%H:%M:%S')}] foto enviada -> {resp.status_code} {resp.json()}")
            except requests.RequestException as e:
                print(f"Error enviando foto: {e}")

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q o ESC
            break
        if key in (ord("n"), ord(" ")):  # n o espacio
            try:
                resp = requests.post(next_turn_url, timeout=5)
                print(f"Cambio de turno -> {resp.json()}")
            except requests.RequestException as e:
                print(f"Error cambiando de turno: {e}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
