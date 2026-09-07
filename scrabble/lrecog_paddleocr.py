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

# Dígitos de multiplicador impresos en el propio tablero (1, 2, 3) - no son
# fichas reales; se descartan explícitamente tras leerlos (ver
# lrecog_easyocr.py para la explicación completa de por qué).
MULTIPLIER_DIGITS = "123"

# Correcciones de dígitos que el OCR confunde con letras visualmente
# parecidas (fichas reales mal leídas como dígito). Se aplican ANTES del
# filtro de dígitos de multiplicador de arriba - por eso el "1" acá deja de
# tratarse como dígito de multiplicador y pasa a interpretarse siempre
# como "L".
DIGIT_TO_LETTER_CORRECTIONS = {
    "5": "S",
    "1": "I",
    "0": "O",
}

# Alfabeto válido de Scrabble en español (incluye Ñ).
VALID_LETTERS = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"

# Umbral mínimo de confianza (0.0-1.0), en la misma escala que usan los
# demás motores (ver OCR_MIN_CONFIDENCE en lrecog_easyocr.py).
OCR_MIN_CONFIDENCE = 0.30

_ocr = None


def get_ocr_reader():
    """
    Crea (una sola vez) la instancia de PaddleOCR - es costosa de
    inicializar (carga los pesos del modelo), igual que EasyOCR.
    """
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        # API 3.x: 'show_log' ya no existe como argumento (eliminado), y
        # 'use_angle_cls' se renombró a 'use_textline_orientation'. No
        # necesitamos clasificación de orientación porque el tablero ya
        # llega enderezado por la homografía ArUco - se desactiva por
        # velocidad. lang="es" selecciona los modelos de
        # detección/reconocimiento apropiados para español.
        #
        # enable_mkldnn=False: desactiva la aceleración Intel MKL-DNN. En
        # algunas CPUs/versiones de Paddle, MKL-DNN da errores (ej.
        # NotImplementedError con ciertas operaciones) o resulta menos
        # predecible que la ruta estándar - se desactiva para priorizar
        # estabilidad sobre la posible ganancia de velocidad.
        # cpu_threads=1: limita la librería matemática de CPU a un solo
        # hilo/núcleo (el default de PaddleOCR es 10), para no competir
        # por CPU con el resto del proceso (ej. el hilo principal de
        # verify_alignment.py dibujando la vista previa).
        # use_doc_orientation_classify / use_doc_unwarping = False: el
        # pipeline moderno de PaddleOCR corre por defecto pasos de
        # preprocesamiento de "documento" pensados para fotos de papel
        # curvado/rotado (ej. una página de libro fotografiada de lado).
        # Nuestra imagen ya es un tablero PLANO y corregido por la
        # homografía ArUco - aplicarle una corrección de curvatura que no
        # necesita desplaza los cuadros detectados de forma importante,
        # más marcado cerca de los bordes superior/inferior (el patrón
        # típico de una corrección de curvatura de página aplicada donde
        # no hace falta).
        _ocr = PaddleOCR(
            lang="es",
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
            cpu_threads=1,
        )
    return _ocr


def _distribuir_texto_en_celdas(texto, bbox, cell_size, board_size, grid, is_occupied_fn):
    """
    Reparte los caracteres de un texto detectado (que puede ser una
    palabra completa de varias fichas) a lo largo del ancho de su cuadro
    delimitador, asignando cada carácter a la casilla que le corresponde
    por posición horizontal. Ver la limitación de granularidad explicada
    en el docstring del módulo.
    """
    # Corrige dígitos que el OCR confunde con letras parecidas (ej. "5"
    # leído en vez de "S") ANTES de filtrar - así "1", "5" y "0" pasan a
    # ser "L", "S", "O" respectivamente, y ya no se tratan como dígitos de
    # multiplicador a descartar.
    texto = texto.upper()
    texto = "".join(DIGIT_TO_LETTER_CORRECTIONS.get(ch, ch) for ch in texto)

    # Filtra a solo letras (incluye Ñ) o dígitos de multiplicador -
    # descarta cualquier símbolo/ruido fuera de ese alfabeto.
    texto = "".join(ch for ch in texto if ch.isalpha() or ch in MULTIPLIER_DIGITS)
    n = len(texto)
    if n == 0:
        return

    # dt_polys de PaddleOCR ya viene en coordenadas con origen
    # arriba-izquierda (igual que la imagen de entrada) - a diferencia de
    # Tesseract, acá NO hace falta invertir el eje Y.
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

        if is_occupied_fn is not None and not is_occupied_fn(row, col):
            continue  # filtro de seguridad: la casilla no parece ocupada

        grid[row][col] = ch


def recognize_board(warped_gray, cell_size, board_size, is_occupied_fn=None, warped_color=None):
    """
    Corre PaddleOCR UNA SOLA VEZ sobre el tablero completo ya enderezado
    y asigna cada letra detectada a su casilla (repartiendo palabras
    completas entre varias casillas si hace falta - ver la limitación de
    granularidad en el docstring del módulo).

    warped_color: el tablero enderezado A COLOR (misma homografía que
    warped_gray, sin pasar por escala de grises). PaddleOCR se beneficia
    de color real de verdad, no solo de cumplir el requisito técnico de
    "3 canales" - por eso, si se provee, se usa directamente en vez de
    replicar el valor de gris en los 3 canales (que técnicamente cumple
    la forma esperada por el modelo, pero no aporta información de color
    real). Si no se provee (por ejemplo, llamando a esta función de forma
    aislada sin pasar por scrabble_processor.read_board_from_warped), se
    cae al reemplazo de gris->BGR como antes.

    Devuelve una matriz board_size x board_size con la letra en cada
    casilla detectada, o None en el resto. Nunca lanza una excepción hacia
    el llamador - si PaddleOCR falla, se devuelve una rejilla vacía en vez
    de tumbar el procesamiento del resto del tablero.
    """
    grid = [[None] * board_size for _ in range(board_size)]

    reader = get_ocr_reader()
    if warped_color is not None:
        imagen_para_ocr = warped_color
    else:
        # Fallback: no hay color real disponible - se replica el gris en
        # los 3 canales solo para cumplir la forma que espera el modelo.
        imagen_para_ocr = cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)

    try:
        results = reader.predict(imagen_para_ocr)
    except Exception as e:
        print(f"[lrecog_paddleocr] PaddleOCR falló leyendo el tablero: {e}")
        return grid

    for res in results:
        textos = res.get("rec_texts", [])
        cajas = res.get("dt_polys", [])
        puntajes = res.get("rec_scores", [1.0] * len(textos))

        for texto, bbox, score in zip(textos, cajas, puntajes):
            if not texto or score < OCR_MIN_CONFIDENCE:
                continue
            _distribuir_texto_en_celdas(
                texto, bbox, cell_size, board_size, grid, is_occupied_fn,
            )

    return grid
