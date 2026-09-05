# -*- coding: utf-8 -*-
"""
Reconocimiento de la letra de una ficha de Scrabble mediante GNU Ocrad
(binario de línea de comandos, invocado por subprocess).

Requiere tener 'ocrad' instalado y en el PATH del sistema:
    sudo apt install ocrad      # Ubuntu / Debian
    brew install ocrad          # macOS

Misma interfaz que lrecog_easyocr.py (get_ocr_reader, recognize_letter),
para que scrabble_processor.py pueda intercambiar el motor de OCR sin
tocar el resto del pipeline de visión.
"""

import shutil
import subprocess

import cv2

# Dígitos de multiplicador impresos en el propio tablero (1, 2, 3) - no son
# fichas reales; se descartan explícitamente tras leerlos (ver
# lrecog_easyocr.py para la explicación completa de por qué).
MULTIPLIER_DIGITS = "123"

# Charset de Ocrad que incluye la Ñ (necesaria para el Scrabble en español;
# 'ascii', el charset por defecto usado en un primer borrador de este
# código, NO la incluye). Verifica los nombres válidos en tu instalación
# con: ocrad --charset=help
OCRAD_CHARSET = "iso-8859-15"

# Ocrad recomienda caracteres de al menos ~20px de alto para reconocer bien
# (ver su manual). Se usa el doble como margen de seguridad, ya que el
# recorte sale de una celda pequeña (~40px) del tablero ya enderezado.
OCRAD_MIN_HEIGHT = 40


def get_ocr_reader():
    """
    Ocrad no tiene un "modelo" que cargar en memoria como EasyOCR - es un
    binario de línea de comandos que se invoca por subprocess en cada
    llamada a recognize_letter(). Esta función solo verifica, una vez al
    arrancar, que el binario esté disponible en el PATH -> falla rápido
    con un mensaje claro en vez de descubrirlo recién en medio de una
    partida.
    """
    if shutil.which("ocrad") is None:
        raise RuntimeError(
            "No se encontró el binario 'ocrad' en el PATH. Instálalo con "
            "tu gestor de paquetes (ej: 'sudo apt install ocrad' en "
            "Ubuntu/Debian, 'brew install ocrad' en macOS) antes de usar "
            "este motor de OCR."
        )
    return True


def _recortar_elemento_central(imagen_gray):
    """
    Aísla el elemento (letra/dígito) más cercano al centro de la celda,
    recortando el resto. Devuelve None si no se encontró ningún contorno
    (celda sin nada reconocible, por ejemplo ruido o una casilla vacía mal
    detectada como ocupada).
    """
    if imagen_gray.ndim != 2:
        imagen_gray = cv2.cvtColor(imagen_gray, cv2.COLOR_BGR2GRAY)

    # Binarización (Otsu) para aislar el elemento del fondo. Se invierte
    # para que el carácter quede blanco sobre fondo negro.
    _, binarizada = cv2.threshold(imagen_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    h_img, w_img = imagen_gray.shape
    centro_img = (w_img / 2, h_img / 2)

    def distancia_al_centro(c):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            return (cX - centro_img[0]) ** 2 + (cY - centro_img[1]) ** 2
        return float("inf")

    contorno_central = min(contornos, key=distancia_al_centro)
    x, y, w, h = cv2.boundingRect(contorno_central)
    if w == 0 or h == 0:
        return None

    return imagen_gray[y:y + h, x:x + w]


def _preparar_para_ocrad(recorte):
    """Escala el recorte si quedó más chico que el mínimo recomendado por
    Ocrad, y le agrega un margen blanco alrededor (ayuda a separar el
    carácter del borde de la imagen)."""
    h, w = recorte.shape[:2]
    if h < OCRAD_MIN_HEIGHT:
        factor = OCRAD_MIN_HEIGHT / float(h)
        recorte = cv2.resize(
            recorte, (max(1, int(w * factor)), OCRAD_MIN_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )

    pad = max(4, int(recorte.shape[0] * 0.15))
    recorte = cv2.copyMakeBorder(
        recorte, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255,
    )
    return recorte


def _ejecutar_ocrad(recorte_gray):
    """
    Codifica el recorte en formato PGM y lo pasa a Ocrad por stdin.
    Devuelve el texto crudo reconocido, o None si algo falló (imagen no
    codificable, binario no encontrado, o Ocrad terminó con error) - nunca
    lanza una excepción hacia el llamador, para no tumbar el procesamiento
    de las otras 224 casillas del tablero por una celda problemática.
    """
    exito, buffer = cv2.imencode(".pgm", recorte_gray)
    if not exito:
        return None

    try:
        proceso = subprocess.run(
            ["ocrad", f"--charset={OCRAD_CHARSET}", "--format=utf8", "-"],
            input=buffer.tobytes(),
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        print(f"[lrecog_ocrad] Ocrad falló en esta celda, se ignora: {e}")
        return None

    texto = proceso.stdout.decode("utf-8", errors="ignore").strip()
    if not texto or texto == "_":
        return None
    return texto


def recognize_letter(cell_gray):
    """
    Lee la letra de una celda con Ocrad. Devuelve (letra, confianza).

    Ocrad no da un score de confianza real (a diferencia de EasyOCR); se
    devuelve 1.0 para cualquier lectura exitosa y 0.0 si no hubo lectura,
    solo para mantener la misma forma de retorno que lrecog_easyocr.py.

    Devuelve (None, 0.0) si:
      - la celda no tiene ningún contorno reconocible,
      - Ocrad no pudo leer nada, o
      - lo leído es en realidad un dígito de multiplicador (1, 2 o 3)
        impreso en el propio tablero, no una ficha real -> se ignora.
    """
    recorte = _recortar_elemento_central(cell_gray)
    if recorte is None:
        return None, 0.0

    recorte = _preparar_para_ocrad(recorte)

    texto = _ejecutar_ocrad(recorte)
    if texto is None:
        return None, 0.0

    # Post-filtrado: solo letras (incluye Ñ) o dígitos de multiplicador,
    # descartando cualquier otro símbolo/ruido que Ocrad haya "adivinado".
    texto = "".join(ch for ch in texto.upper() if ch.isalpha() or ch in MULTIPLIER_DIGITS)
    if not texto:
        return None, 0.0

    if all(ch in MULTIPLIER_DIGITS for ch in texto):
        return None, 1.0

    return texto, 1.0
