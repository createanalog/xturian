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

# --psm 10 = tratar la imagen completa como UN SOLO carácter (ideal acá,
# ya que cada celda trae como mucho una letra o un dígrafo).
# --oem 3  = motor por defecto (LSTM + legacy combinados).
TESSERACT_CONFIG = (
    "--psm 10 --oem 3 "
    f"-c tessedit_char_whitelist={TESSERACT_WHITELIST}"
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
