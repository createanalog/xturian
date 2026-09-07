# -*- coding: utf-8 -*-
"""
Reconocimiento de la letra de una ficha de Scrabble mediante Tesseract OCR
(vía la librería pytesseract).

Requiere el motor de Tesseract instalado en el sistema (no solo el paquete
de Python) con el modelo de idioma español:
    sudo apt install tesseract-ocr tesseract-ocr-spa   # Ubuntu / Debian
    brew install tesseract tesseract-lang               # macOS

Y el wrapper de Python:
    pip install pytesseract

Misma interfaz que lrecog_easyocr.py / lrecog_ocrad.py (get_ocr_reader,
recognize_letter), para que scrabble_processor.py pueda intercambiar el
motor de OCR sin tocar el resto del pipeline de visión.
"""

import shutil

import cv2
import pytesseract

# Dígitos de multiplicador impresos en el propio tablero (1, 2, 3) - no son
# fichas reales; se descartan explícitamente tras leerlos (ver
# lrecog_easyocr.py para la explicación completa de por qué).
MULTIPLIER_DIGITS = "123"

# Alfabeto válido de Scrabble en español (incluye Ñ).
VALID_LETTERS = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"

# Tesseract permite restringir directamente los caracteres que puede
# reconocer vía tessedit_char_whitelist - equivalente al allowlist que
# usa EasyOCR.
TESSERACT_WHITELIST = VALID_LETTERS + MULTIPLIER_DIGITS

# --psm 10 = tratar la imagen completa como UN SOLO carácter (para
# recognize_letter(), que sigue disponible como fallback celda-por-celda).
# --oem 3  = motor por defecto (LSTM + legacy combinados).
# thresholding_method=0 = fuerza el Otsu clásico (el más simple/rápido) en
# vez del método adaptativo de Tesseract 5 - un ajuste menor, ya que de
# todos modos le entregamos una imagen pre-binarizada nosotros mismos (ver
# _binarizar más abajo), así que ese paso interno ya es casi instantáneo.
TESSERACT_CONFIG = (
    "--psm 10 --oem 3 "
    f"-c tessedit_char_whitelist={TESSERACT_WHITELIST} "
    "-c thresholding_method=0"
)

# --psm 11 = "texto disperso, sin orden particular" - para recognize_board(),
# que procesa el tablero COMPLETO de una sola vez: encaja con letras
# aisladas repartidas por 225 casillas, a diferencia de un documento con
# líneas/párrafos continuos (que es lo que psm 10/6/3 asumen).
#
# Fracción de la altura de una celda que se usa como piso de x-height
# creíble (textord_min_xheight, en píxeles) - filtra ruido (motas de una
# casilla de color, artefactos JPEG, restos de la rejilla) mucho más chico
# que una letra real, sin descartar letras genuinas.
MIN_XHEIGHT_CELL_FRACTION = 0.35


def _build_board_ocr_config(cell_size):
    min_xheight = max(1, int(cell_size * MIN_XHEIGHT_CELL_FRACTION))
    return (
        "--psm 11 --oem 3 "
        f"-c tessedit_char_whitelist={TESSERACT_WHITELIST} "
        "-c thresholding_method=0 "
        f"-c textord_min_xheight={min_xheight}"
    )

# Modelo de idioma de Tesseract. 'spa' reconoce mejor los caracteres del
# español (en particular la Ñ) que el modelo en inglés por defecto.
TESSERACT_LANG = "spa"

# pytesseract reporta confianza en escala 0-100; este umbral está en esa
# misma escala (se normaliza a 0.0-1.0 recién al devolver el resultado,
# para mantener la misma forma de retorno que lrecog_easyocr.py).
OCR_MIN_CONFIDENCE_PCT = 30.0


def get_ocr_reader():
    """
    Tesseract no se "carga" en memoria como un modelo de EasyOCR - es un
    binario del sistema invocado por pytesseract en cada llamada. Esta
    función solo verifica, una vez al arrancar, que el binario y el
    modelo de idioma español estén disponibles -> falla rápido con un
    mensaje claro en vez de descubrirlo recién en medio de una partida.
    """
    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "No se encontró el binario 'tesseract' en el PATH. Instálalo "
            "con tu gestor de paquetes (ej: 'sudo apt install tesseract-ocr "
            "tesseract-ocr-spa' en Ubuntu/Debian, 'brew install tesseract "
            "tesseract-lang' en macOS) antes de usar este motor de OCR."
        )

    try:
        idiomas_disponibles = pytesseract.get_languages(config="")
    except Exception:
        idiomas_disponibles = []

    if TESSERACT_LANG not in idiomas_disponibles:
        print(
            f"[lrecog_tesseract] AVISO: el modelo de idioma '{TESSERACT_LANG}' "
            f"no está instalado (idiomas detectados: {idiomas_disponibles}). "
            f"Instálalo con 'sudo apt install tesseract-ocr-spa' o "
            f"equivalente - mientras tanto Tesseract puede fallar en "
            f"reconocer la Ñ correctamente."
        )
    return True


def recognize_letter(cell_gray):
    """
    Lee la letra de una celda con Tesseract. Devuelve (letra, confianza),
    con la confianza normalizada a 0.0-1.0 (Tesseract internamente reporta
    0-100).

    Devuelve (None, score) si:
      - no hay lectura suficientemente confiable, o
      - lo leído es en realidad un dígito de multiplicador (1, 2 o 3)
        impreso en el propio tablero, no una ficha real -> se ignora.
    """
    # Igual que con EasyOCR, Tesseract trabaja mejor sobre imágenes más
    # grandes que una celda recortada de ~30-40px.
    upscale_size = 128
    cell_big = cv2.resize(cell_gray, (upscale_size, upscale_size), interpolation=cv2.INTER_CUBIC)

    # Binarización (Otsu): con --psm 10 (un solo carácter), ayuda a
    # Tesseract a separar la letra del color de fondo de la casilla.
    _, cell_bin = cv2.threshold(cell_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        data = pytesseract.image_to_data(
            cell_bin,
            lang=TESSERACT_LANG,
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractError as e:
        print(f"[lrecog_tesseract] Tesseract falló en esta celda, se ignora: {e}")
        return None, 0.0

    # image_to_data devuelve una fila por cada bloque/palabra detectado
    # (casi siempre una sola con --psm 10); nos quedamos con la de mayor
    # confianza que tenga texto no vacío.
    mejor_texto, mejor_conf = None, -1.0
    for texto, conf_str in zip(data.get("text", []), data.get("conf", [])):
        texto = texto.strip()
        try:
            conf = float(conf_str)
        except (TypeError, ValueError):
            continue
        if texto and conf > mejor_conf:
            mejor_texto, mejor_conf = texto, conf

    if mejor_texto is None or mejor_conf < 0:
        return None, 0.0

    confidence = max(0.0, min(1.0, mejor_conf / 100.0))
    if mejor_conf < OCR_MIN_CONFIDENCE_PCT:
        return None, confidence

    # Post-filtrado defensivo: solo letras (incluye Ñ) o dígitos de
    # multiplicador, por si el whitelist no bastó para descartar ruido.
    texto = "".join(ch for ch in mejor_texto.upper() if ch.isalpha() or ch in MULTIPLIER_DIGITS)
    if not texto:
        return None, confidence

    if all(ch in MULTIPLIER_DIGITS for ch in texto):
        return None, confidence

    return texto, confidence


def recognize_board(warped_gray, cell_size, board_size, is_occupied_fn=None, warped_color=None):
    """
    Corre Tesseract UNA SOLA VEZ sobre el tablero completo ya enderezado,
    en vez de una vez por cada una de las 225 casillas. Esto paga el costo
    fijo de arrancar el proceso y cargar el modelo de idioma UNA sola vez
    por lectura, en vez de 20-25 veces — que es lo que causaba los ~10
    segundos de demora con el enfoque celda por celda.

    Cada carácter detectado se asigna a la casilla que lo contiene, según
    la posición de su centro (image_to_boxes da la posición en píxeles de
    cada carácter individual, con origen abajo-izquierda - se convierte a
    origen arriba-izquierda antes de mapear a fila/columna).

    is_occupied_fn(row, col) -> bool es opcional: si se pasa, se usa como
    filtro de seguridad adicional para descartar ruido que caiga en una
    casilla que el chequeo de ocupación (diff contra el tablero vacío) dice
    que está vacía - reduce falsos positivos sin cambiar la idea central
    (una sola pasada de OCR + asignación por posición).

    warped_color se acepta mas NO se usa - Tesseract trabaja en escala de
    grises y no se beneficia de color real. Solo está en la firma para que
    la interfaz sea intercambiable con otros motores (ej. PaddleOCR) que
    sí lo necesitan - ver scrabble_processor.read_board_from_warped().

    Se le indica a Tesseract un x-height mínimo creíble (textord_min_xheight,
    ver MIN_XHEIGHT_CELL_FRACTION) proporcional a cell_size, para que
    descarte de entrada trazos mucho más chicos que una letra real sin
    tener que reconocerlos primero.

    Devuelve una matriz board_size x board_size con la letra en cada
    casilla detectada, o None en el resto.
    """
    h, w = warped_gray.shape[:2]

    # Binarización global, una sola vez para todo el tablero. Las FICHAS
    # REALES quedan siempre como letra oscura sobre ficha clara sin
    # importar el color de la casilla impresa debajo (la ficha física cubre
    # por completo el color del tablero) - así que la polaridad "negro
    # sobre blanco" se mantiene consistente para lo único que de verdad
    # importa: las fichas jugadas. Las casillas vacías con dígitos de
    # multiplicador impresos en colores distintos pueden binarizar distinto,
    # pero da igual: esos dígitos se ignoran de todos modos.
    _, board_bin = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        raw_boxes = pytesseract.image_to_boxes(
            board_bin, lang=TESSERACT_LANG, config=_build_board_ocr_config(cell_size),
        )
    except pytesseract.TesseractError as e:
        print(f"[lrecog_tesseract] Tesseract falló leyendo el tablero completo: {e}")
        return [[None] * board_size for _ in range(board_size)]

    grid = [[None] * board_size for _ in range(board_size)]

    for line in raw_boxes.splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue  # línea con formato inesperado, se ignora

        ch, left, bottom, right, top, _page = parts
        ch = ch.upper()
        if not (ch.isalpha() or ch in MULTIPLIER_DIGITS):
            continue  # símbolo/ruido fuera del alfabeto esperado

        left, bottom, right, top = int(left), int(bottom), int(right), int(top)
        cx = (left + right) / 2.0
        # image_to_boxes usa origen abajo-izquierda; se convierte a
        # arriba-izquierda (el mismo sistema que usa el resto del pipeline).
        cy = h - (top + bottom) / 2.0

        col = int(cx // cell_size)
        row = int(cy // cell_size)
        if not (0 <= row < board_size and 0 <= col < board_size):
            continue  # cayó justo en el borde/fuera del tablero, se ignora

        if ch in MULTIPLIER_DIGITS:
            continue  # dígito de multiplicador del tablero -> se ignora

        if is_occupied_fn is not None and not is_occupied_fn(row, col):
            continue  # filtro de seguridad: la casilla no parece ocupada

        grid[row][col] = ch

    return grid
