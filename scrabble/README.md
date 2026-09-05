# Scrabble por visión con ESP32-CAM

Sistema que usa una ESP32-CAM para fotografiar el tablero de Scrabble
periódicamente, detecta cada jugada, calcula la puntuación (reglas de
Scrabble en español) y lleva el acumulado por jugador.

## Arquitectura

```
ESP32-CAM  --(foto JPEG cada 3s por WiFi)-->  server.py (tu PC)
                                                  |
                                                  v
                                    detecta tablero (ArUco) -> lee letras
                                    (EasyOCR) -> compara con la
                                    jugada anterior -> calcula puntos
                                                  |
                                                  v
                                     panel web http://TU_PC:5000/
```

La ESP32-CAM solo captura y envía fotos; todo el procesamiento (que requiere
más potencia de la que tiene la placa) ocurre en tu ordenador.

## 1. Instalación

El reconocimiento de letras es intercambiable entre **tres motores de OCR**
distintos, elegibles con `--ocr-engine {tesseract,easyocr,ocrad}` al arrancar
`server.py` o `verify_alignment.py`. **Por defecto se usa `tesseract`** si no
especificas nada. Solo necesitas instalar el motor que realmente vayas a usar
(la carga de cada uno es perezosa - no hace falta tener los 3 instalados).

Dependencias base (siempre necesarias, sin importar el motor de OCR):
```bash
pip install flask opencv-contrib-python numpy
```

**Tesseract (motor por defecto):**
```bash
sudo apt install tesseract-ocr tesseract-ocr-spa   # Ubuntu/Debian
brew install tesseract tesseract-lang               # macOS
pip install pytesseract
```
El modelo de idioma español (`tesseract-ocr-spa` / `tesseract-lang`) es
importante — sin él, Tesseract reconoce peor la Ñ.

**EasyOCR:**
```bash
pip install easyocr
```
Notas:
- La primera vez que se use, EasyOCR descargará los pesos de su modelo
  (~decenas de MB) — necesitas conexión a internet esa primera vez.
- Corre en CPU por defecto y funciona bien así; si tienes GPU CUDA
  disponible, puedes activarla cambiando `gpu=False` a `gpu=True` en
  `get_ocr_reader()` dentro de `lrecog_easyocr.py` para mayor velocidad.
- **Uso de CPU:** por defecto, PyTorch (la base de EasyOCR) intenta usar
  todos los núcleos disponibles en cada inferencia, lo que puede saturar
  el CPU. El código ya limita esto con `torch.set_num_threads(2)` al
  inicio de `lrecog_easyocr.py` — ajusta ese número según tu máquina
  si quieres dejar más o menos núcleos libres para el resto del sistema.

**Ocrad:**
```bash
sudo apt install ocrad   # Ubuntu/Debian
brew install ocrad        # macOS
```
No necesita ningún paquete de Python adicional — se invoca como binario
de línea de comandos.

Nota general: usa `opencv-contrib-python` (no `opencv-python` a secas) para
tener disponible el módulo `cv2.aruco`.


## 2. Preparación física del tablero

1. Imprime 4 marcadores ArUco del diccionario `DICT_4X4_50`, con ids **0, 1,
   2 y 3**. Puedes generarlos con:
   ```python
   import cv2
   d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
   for i in range(4):
       img = cv2.aruco.generateImageMarker(d, i, 200)
       cv2.imwrite(f"marker_{i}.png", img)
   ```
2. Pega los marcadores justo fuera de las 4 esquinas del tablero:
   - id 0 -> esquina superior izquierda
   - id 1 -> esquina superior derecha
   - id 2 -> esquina inferior derecha
   - id 3 -> esquina inferior izquierda
3. Monta la ESP32-CAM en un soporte fijo, mirando el tablero desde arriba,
   con buena luz constante (evita sombras móviles de las manos si es posible).
4. **Verifica la alineación y el OCR antes de calibrar/jugar:** con la
   webcam apuntando al tablero, ejecuta
   ```bash
   python verify_alignment.py --camera 0
   ```
   Se abre una ventana en vivo con el tablero enderezado, la rejilla 15x15
   dibujada, y la letra reconocida por OCR escrita en **verde brillante**
   sobre cada ficha detectada. Si la rejilla no coincide con los bordes
   reales de las casillas, reajusta la posición de los marcadores; si las
   letras no se leen bien, prueba mejorando la luz o el enfoque. Cierra con
   `q` o `ESC`. (Necesita que ya hayas calibrado el tablero vacío, ver
   el paso 5 más abajo.)

## 3. Botón físico de cambio de turno

Conecta un pulsador simple entre **GPIO13 y GND** (usa la resistencia pull-up
interna del ESP32, no necesitas nada más). Al pulsarlo, la placa avisa al
servidor y este pasa el turno al siguiente jugador — útil cuando alguien pasa
turno, cambia fichas, o quieres corregir manualmente la rotación automática.

También puedes disparar el mismo cambio de turno sin el botón, llamando
directamente al endpoint desde cualquier navegador o `curl`:
```bash
curl -X POST http://TU_PC:5000/next_turn
```

## 4. Configurar y flashear la ESP32-CAM

Edita en `esp32_cam_capture.ino`:
- `WIFI_SSID` / `WIFI_PASSWORD`
- `SERVER_URL` con la IP de tu PC en la red local (ej: `http://192.168.1.100:5000/upload`)

Flashea con el Arduino IDE (board "AI Thinker ESP32-CAM", GPIO0 a GND para
entrar en modo programación).

## 5. Calibración (obligatoria antes de la primera partida)

Con el tablero **vacío** y bien iluminado, toma una foto (puedes usar la
propia ESP32-CAM apuntando a `http://IP_ESP32/capture` si añades ese modo, o
simplemente una foto con el móvil desde el mismo ángulo) y ejecuta:

```bash
python calibrate.py empty foto_tablero_vacio.jpg
```

Con eso basta — el reconocimiento de letras usa OCR (Tesseract, EasyOCR u
Ocrad, según elijas), que **no necesita** plantillas ni fotos previas por
cada letra.

## 6. Arrancar el servidor

```bash
python server.py --players 2
python server.py --players 2 --ocr-engine easyocr   # o --ocr-engine ocrad
```

Si no especificas `--ocr-engine`, se usa **Tesseract** por defecto.

Abre `http://TU_PC:5000/` para ver el marcador en vivo. Los turnos se asignan
de forma automática y rotativa (jugador 1, 2, 3... y vuelve a empezar) cada
vez que se detecta una jugada válida y estable.

## Limitaciones importantes (léelo antes de confiar en el marcador)

- **El reconocimiento de letras es el eslabón más débil, y varía según el
  motor de OCR elegido.** Tesseract (el default) y EasyOCR son más
  tolerantes al ruido/blur/iluminación que Ocrad, pero ninguno es
  perfecto — cualquiera puede confundir letras parecidas o fallar con
  fotos muy desenfocadas. Si uno rinde mal con tu tablero/iluminación,
  prueba con `--ocr-engine` cambiando de motor antes de asumir que hay
  que ajustar la calibración física. Revisa el panel web tras cada jugada.
- **Los dígitos de multiplicador impresos en el propio tablero (1, 2, 3)
  se ignoran automáticamente.** Si `cell_is_occupied()` marca por error una
  casilla vacía como ocupada (por ejemplo, por una sombra), el OCR puede
  leer ese dígito de fondo — pero como está en el allowlist, se reconoce
  correctamente como dígito y se descarta en vez de forzarse a la letra
  más parecida. Esas casillas simplemente no generan ninguna ficha nueva.
- Los dígrafos del set clásico en español (**CH, LL, RR**) se leen como
  texto de 2 caracteres de forma natural; si tu edición no los tiene como
  fichas propias, no afecta en nada — simplemente no aparecerán.
- Si el sistema no reconoce una letra con confianza suficiente, **no
  puntúa esa jugada** (verás `letras_no_reconocidas` en la consola) en vez
  de arriesgarse a un cálculo erróneo. Puedes forzar el reintento moviendo
  ligeramente la cámara/luz y esperando la siguiente foto.
- No se valida que la palabra exista en el diccionario español ni las
  reglas de conexión con fichas ya en el tablero más allá de la alineación;
  asume buena fe de los jugadores.
- Los valores de ficha en `board_config.py` corresponden a la edición
  clásica en español (con CH, LL, RR, Ñ como fichas independientes). Si tu
  edición es distinta, edita ese diccionario.
- **La calibración de marcadores se cachea entre frames.** Si en un frame
  puntual no se detectan los 4 marcadores (mano tapando uno, motion blur),
  el sistema reutiliza automáticamente la última detección válida en vez
  de descartar ese frame. Esto lo hace más tolerante a oclusiones breves,
  pero asume que el tablero no se movió durante ese lapso — si el tablero
  se mueve justo mientras los marcadores están tapados, la calibración
  cacheada queda desactualizada hasta la siguiente detección real.
- El turno se asigna por rotación automática tras cada jugada válida, pero
  puedes forzar el cambio en cualquier momento con el botón físico (GPIO13)
  o llamando a `POST /next_turn` — útil si un jugador pasa turno o cambia
  fichas sin jugar.
