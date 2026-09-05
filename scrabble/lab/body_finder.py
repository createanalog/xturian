import sys
import os
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

def recortar_elemento_central(nombre_imagen: str):
    # 1. Comprobar que la imagen exista
    if not os.path.exists(nombre_imagen):
        print(f"Error: El archivo '{nombre_imagen}' no existe.")
        return

    # 2. Cargar la imagen
    imagen = cv2.imread(nombre_imagen)
    
    kernel_sharpen = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    #imagen = cv2.GaussianBlur(imagen, (5, 5), 0)
    #imagen = cv2.bilateralFilter(imagen, d=9, sigmaColor=75, sigmaSpace=75)
    alpha = 2.0  # Factor de contraste
    beta = -50   # Ajuste opcional de brillo para compensar
    #imagen = cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)
    #imagen = cv2.filter2D(imagen, -1, kernel_sharpen)

    imagen= cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = preprocesar_para_ocrad(imagen)
    gris = imagen
    # 3. Binarización (Umbralización OTSU) para aislar el elemento del fondo
    # Invertimos (THRESH_BINARY_INV) para que el carácter sea blanco sobre fondo negro
    _, binarizada = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Encontrar contornos
    contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        print("No se encontró ningún elemento en la imagen.")
        return

    # 5. Seleccionar el contorno más cercano al centro de la imagen
    h_img, w_img = gris.shape
    centro_img = (w_img / 2, h_img / 2)
    
    def distancia_al_centro(c):
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX - centro_img[0])**2 + (cY - centro_img[1])**2
        return float("inf")

    contorno_central = min(contornos, key=distancia_al_centro)

    # 6. Obtener el rectángulo delimitador (ROI) y recortar
    x, y, w, h = cv2.boundingRect(contorno_central)
    recorte = gris[y:y+h, x:x+w]

    # 7. Generar el nombre de salida con "_trimmed"
    nombre_base, extension = os.path.splitext(nombre_imagen)
    nombre_salida = f"{nombre_base}_trimmed{extension}"

    # CLAHE filter just before save
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))    
    #recorte = clahe.apply(recorte)
    
    # 8. Guardar la imagen recortada
    cv2.imwrite(nombre_salida, recorte)
    print(f"Imagen procesada guardada como: '{nombre_salida}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python recortar.py <nombre_imagen>")
        sys.exit(1)

    recortar_elemento_central(sys.argv[1])
    
