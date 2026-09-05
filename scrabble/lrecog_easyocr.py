# -*- coding: utf-8 -*-
"""
Reconocimiento de la letra de una ficha de Scrabble mediante EasyOCR.

Aislado en su propio módulo para que sea fácil de reemplazar por otro
motor de reconocimiento en el futuro sin tocar el resto del pipeline de
visión (detección de marcadores, perspectiva, puntuación) en
scrabble_processor.py.
"""

import cv2
import torch

# Limita cuántos hilos usa PyTorch para inferencia en CPU. Por defecto,
# PyTorch intenta usar todos los núcleos disponibles en cada llamada al
# modelo de EasyOCR, lo que puede saturar el CPU (los 4 núcleos al 100%)
# incluso en máquinas de escritorio normales. Ajusta este número según tu
# CPU: déjalo más bajo si quieres dejar núcleos libres para el resto del
# sistema (por ejemplo, mientras corre el bucle de la webcam en paralelo).
torch.set_num_threads(2)

import easyocr

OCR_MIN_CONFIDENCE = 0.30      # umbral mínimo de confianza de EasyOCR

# Letras válidas del alfabeto español de Scrabble. Restringir el OCR a este
# conjunto reduce muchísimo los falsos positivos (evita que "adivine"
# caracteres que no pueden aparecer en una ficha).
VALID_LETTERS = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"

# El propio tablero (no las fichas) tiene impresos los dígitos 1, 2 y 3 en
# las casillas de multiplicador. Si el OCR solo tuviera letras en su
# allowlist, un dígito detectado se vería FORZADO a interpretarse como la
# letra más parecida (ej: un "2" leído como "Z"), en vez de fallar limpio.
# Por eso se agregan al allowlist para que el OCR los reconozca tal cual
# son -> y luego se descartan explícitamente en recognize_letter().
MULTIPLIER_DIGITS = "123"
OCR_ALLOWLIST = VALID_LETTERS + MULTIPLIER_DIGITS

# El lector de EasyOCR es costoso de crear (carga los pesos del modelo);
# se instancia UNA sola vez y se reutiliza en cada llamada a recognize_letter.
# gpu=False por defecto para que funcione en cualquier máquina sin CUDA;
# cambia a gpu=True si tienes GPU compatible disponible.
_ocr_reader = None


def get_ocr_reader():
    global _ocr_reader
   # if _ocr_reader is None:
    #    _ocr_reader = easyocr.Reader(["es"], gpu=False)
    return _ocr_reader


def recognize_letter(cell_gray):
    """
    Lee la letra de una celda con EasyOCR. Devuelve (letra, confianza).
    Devuelve (None, score) si:
      - no hay lectura suficientemente confiable, o
      - lo que se leyó es en realidad un dígito de multiplicador (1, 2 o 3)
        impreso en el propio tablero, no una ficha real -> se ignora.
    """
    reader = get_ocr_reader()

    # EasyOCR trabaja mejor sobre imágenes algo más grandes que una celda
    # recortada de ~30-35px; se escala antes de pasarla al modelo.
    upscale_size = 128
    cell_big = cv2.resize(cell_gray, (upscale_size, upscale_size), interpolation=cv2.INTER_CUBIC)
    cell_rgb = cv2.cvtColor(cell_big, cv2.COLOR_GRAY2RGB)

    # min_size filtra regiones de texto candidatas más chicas que esto (en
    # píxeles, dentro del espacio de la imagen ya escalada arriba). Se fija
    # como el 40% de la altura de la celda procesada, para descartar ruido
    # de fondo (líneas de la rejilla, texturas) que sea mucho más chico que
    # una letra real ocupando la casilla.
    min_size = int(upscale_size * 0.4)

    results = reader.readtext(
        cell_rgb,
        allowlist=OCR_ALLOWLIST,
        detail=1,
        paragraph=False,
        min_size=min_size,
        mag_ratio=1,
    )

    if not results:
        return None, 0.0

    # Nos quedamos con la lectura de mayor confianza (debería haber una sola,
    # ya que una ficha trae una única letra/dígrafo).
    _, text, confidence = max(results, key=lambda r: r[2])
    text = text.strip().upper()

    if not text or confidence < OCR_MIN_CONFIDENCE:
        return None, confidence

    # Es un dígito de multiplicador impreso en el tablero (1, 2 o 3), no una
    # ficha real -> se ignora en vez de tratarlo como letra.
    if all(ch in MULTIPLIER_DIGITS for ch in text):
        return None, confidence

    return text, confidence
