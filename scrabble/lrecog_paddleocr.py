# -*- coding: utf-8 -*-
"""
Reconocimiento de la letra de una ficha de Scrabble mediante PaddleOCR.

Requiere:
    pip install paddlepaddle paddleocr

Misma interfaz de recognize_board() que lrecog_tesseract.py, para que
scrabble_processor.py pueda intercambiar el motor de OCR sin tocar el
resto del pipeline de visión.

IMPORTANTE - limitación de granularidad (léela antes de confiar en este
motor para partidas reales):
A diferencia de Tesseract (pytesseract.image_to_boxes da la posición de
cada CARÁCTER individual), PaddleOCR detecta y reconoce texto a nivel de
LÍNEA/PALABRA completa. Si dos o más fichas quedan adyacentes formando una
palabra real (lo normal en cualquier partida en curso), PaddleOCR devuelve
UN SOLO cuadro delimitador con el texto completo de la palabra, no uno por
letra. Este módulo reparte manualmente los caracteres de cada palabra
detectada a lo largo del ancho de su cuadro delimitador (asumiendo
orientación horizontal y caracteres de ancho aproximadamente uniforme) para
poder asignar cada letra a su casilla - una aproximación razonable, pero
menos precisa que la granularidad nativa por carácter de Tesseract,
especialmente si las fichas de una palabra no están perfectamente
alineadas o son de ancho muy distinto entre sí (ej. "I" vs "M").

NOTA SOBRE LA VERSIÓN DE PADDLEOCR: este módulo asume la API 3.x
(ocr.predict(...) devolviendo objetos tipo diccionario con las claves
'rec_texts', 'dt_polys', 'rec_scores'). Si tu instalación es una versión
2.x, el método y el formato de resultado son distintos (ocr.ocr(...)
devolviendo listas anidadas [bbox, (texto, score)]) - verifica tu versión
con 'pip show paddleocr' si algo no coincide.
"""

import cv2

MULTIPLIER_DIGITS = "123"

DIGIT_TO_LETTER_CORRECTIONS = {
    "5": "S",
    "1": "I",
    "0": "O"
}

VALID_LETTERS = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"

OCR_MIN_CONFIDENCE = 0.30

_ocr = None


def get_ocr_reader():

    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
      
        _ocr = PaddleOCR(
            lang="es",
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
            cpu_threads=1,
        )
    return _ocr


def _distribuir_texto_en_celdas(texto, bbox, cell_size, board_size, grid):

    texto = texto.upper()
    texto = "".join(DIGIT_TO_LETTER_CORRECTIONS.get(ch, ch) for ch in texto)

    # Filtra a solo letras (incluye Ñ) o dígitos de multiplicador -
    # descarta cualquier símbolo/ruido fuera de ese alfabeto.
    texto = "".join(ch for ch in texto if ch != "???" and ch.isalpha() or ch in MULTIPLIER_DIGITS)
    n = len(texto)
    if n == 0:
        return

    x_min = float(bbox[:, 0].min())
    x_max = float(bbox[:, 0].max())
    y_center = float(bbox[:, 1].mean())

    ancho_por_caracter = (x_max - x_min) / n

    for i, ch in enumerate(texto):
        cx = x_min + ancho_por_caracter * (i + 0.5)
        col = int(cx // cell_size)
        row = int(y_center // cell_size)

        if not (0 <= row < board_size and 0 <= col < board_size):
            continue  # cayó fuera del tablero, se ignora

        if ch in MULTIPLIER_DIGITS:
            continue  # dígito de multiplicador del tablero -> se ignora

        grid[row][col] = ch


def recognize_board(image, cell_size, board_size):

    grid = [[None] * board_size for _ in range(board_size)]
  
    reader = get_ocr_reader()

    try:
        results = reader.predict(image)
    except Exception as e:
        print(f"[lrecog_paddleocr] PaddleOCR falló leyendo el tablero: {e}")
        return grid
    print("JUST BEFORE")
    if results and results[0] is not None:   ## NO NECESARIO??????
        print("DETECTANDO")
        for res in results:
            textos = res.get("rec_texts", [])
            cajas = res.get("dt_polys", [])
            puntajes = res.get("rec_scores", [1.0] * len(textos))

            for texto, bbox, score in zip(textos, cajas, puntajes):
                if not texto or score < OCR_MIN_CONFIDENCE:
                    continue
                print(f"TEXTO ENCONTRADO: {texto}")
                _distribuir_texto_en_celdas(
                    texto, bbox, cell_size, board_size, grid,
                )
    else:
        print("No text detected")

    return grid
