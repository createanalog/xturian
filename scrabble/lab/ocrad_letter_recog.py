import subprocess
import cv2
import numpy as np

def preprocesar_para_ocrad(imagen_gris):
    # 1. Aumentar el tamaño (Interpolación Cúbica)
    # Al aumentar la resolución, le das a los filtros espaciales más píxeles sobre los que actuar.
    h, w = imagen_gris.shape[:2]
    imagen_ampliada = cv2.resize(imagen_gris, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    # 2. Aumentar el contraste local con CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    imagen_contrastada = clahe.apply(imagen_ampliada)

    # 3. Aplicar Máscara de Desenfoque (Unsharp Masking) para afilar bordes borrosos
    # Resta una versión desenfocada para resaltar las transiciones rápidas de intensidad
    gauss = cv2.GaussianBlur(imagen_contrastada, (0, 0), 3.0)
    imagen_enfocada = cv2.addWeighted(imagen_contrastada, 1.8, gauss, -0.8, 0)

    # 4. Binarización agresiva (Otsu)
    # Convertimos los bordes grises suaves en un trazo negro sólido con fondo blanco puro
    _, binarizada = cv2.threshold(imagen_enfocada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Opcional: Limpieza morfológica rápida para rellenar huecos en el trazo
    kernel = np.ones((3, 3), np.uint8)
    imagen_limpia = cv2.morphologyEx(binarizada, cv2.MORPH_CLOSE, kernel)

    return imagen_limpia

def es_texto_oscuro_sobre_blanco(imagen_gris) -> bool:
    """
    Determina si la imagen representa una letra oscura sobre un fondo claro
    comparando el brillo del perímetro exterior contra el promedio general.
    """
    h, w = imagen_gris.shape[:2]

    # Crear una máscara para extraer el perímetro (borde exterior de 2px)
    mascara_bordes = np.ones((h, w), dtype=bool)
    mascara_bordes[2:h-2, 2:w-2] = False

    # Intensidad promedio del fondo (bordes) y de toda la imagen
    brillo_fondo = np.mean(imagen_gris[mascara_bordes])
    brillo_promedio = np.mean(imagen_gris)

    # Es válido si el fondo es claro (> 127) y más brillante que el promedio global
    return brillo_fondo > 127 and brillo_fondo > brillo_promedio


def recortar_elemento_central(imagen):
    h_img, w_img = imagen.shape[:2]

    # Binarización (OTSU) para aislar el elemento del fondo
    _, binarizada = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar contornos
    contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        print("No se encontró ningún elemento en la imagen.")
        return None

    centro_img = (w_img / 2, h_img / 2)
    
    def distancia_al_centro(c):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX - centro_img[0])**2 + (cY - centro_img[1])**2
        return float("inf")

    contorno_central = min(contornos, key=distancia_al_centro)

    # Obtener rectángulo delimitador (ROI)
    x, y, w, h = cv2.boundingRect(contorno_central)

    # Validación de tamaño según la altura original (mínimo 50%, máximo 80%)
    if not (0.40 * h_img <= h <= 0.90 * h_img):
        print(f"Recorte rechazado por tamaño inválido: h={h}px (debe estar entre {int(0.40*h_img)}px y {int(0.90*h_img)}px) XXX")
        return None

    trimmed = imagen[y:y+h, x:x+w]

    # Validar que la imagen sea de texto oscuro sobre blanco
    if not es_texto_oscuro_sobre_blanco(trimmed):
        print("Imagen descartada: No cumple con el criterio de texto oscuro sobre fondo blanco.")
        return None

    return trimmed


def ocr_desde_matriz_opencv(imagen_cv2):
    # Asegurar que esté en escala de grises
    if len(imagen_cv2.shape) == 3:
        imagen_cv2 = cv2.cvtColor(imagen_cv2, cv2.COLOR_BGR2GRAY)

    recorte = recortar_elemento_central(imagen_cv2)
    if recorte is None:
        return None, 0

    recorte = preprocesar_para_ocrad(recorte);

    # Codificar la imagen en PGM en memoria
    éxito, buffer = cv2.imencode('.pgm', recorte)
    if not éxito:
        raise ValueError("No se pudo codificar la imagen en memoria.")

    # Ejecución de OCRAD mediante subprocess
    try:
        proceso = subprocess.run(
            [
                'ocrad',
                '--charset=iso-8859-15',  # Incluye la 'Ñ' y caracteres extendidos
                '--format=utf8',          # Fuerza salida codificada en UTF-8
                '-'
            ],
            input=buffer.tobytes(),
            capture_output=True,
            text=False,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"OCRAD devolvió un código de error ({e.returncode}).")
        return None, 0

    texto_resultado = proceso.stdout.decode('utf-8').strip()
    
    if not texto_resultado or texto_resultado == '_':
        return None, 1
    else:
        return texto_resultado, 1
