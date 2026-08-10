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
                                    (template matching) -> compara con la
                                    jugada anterior -> calcula puntos
                                                  |
                                                  v
                                     panel web http://TU_PC:5000/
```

La ESP32-CAM solo captura y envía fotos; todo el procesamiento (que requiere
más potencia de la que tiene la placa) ocurre en tu ordenador.

## 1. Instalación

```bash
pip install flask opencv-python numpy
```

Nota: usa `opencv-contrib-python` en vez de `opencv-python` si tu versión de
OpenCV no trae el módulo `cv2.aruco` (algunas instalaciones lo separan).

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

## 3. Configurar y flashear la ESP32-CAM

Edita en `esp32_cam_capture.ino`:
- `WIFI_SSID` / `WIFI_PASSWORD`
- `SERVER_URL` con la IP de tu PC en la red local (ej: `http://192.168.1.100:5000/upload`)

Flashea con el Arduino IDE (board "AI Thinker ESP32-CAM", GPIO0 a GND para
entrar en modo programación).

## 4. Calibración (obligatoria antes de la primera partida)

Con el tablero **vacío** y bien iluminado, toma una foto (puedes usar la
propia ESP32-CAM apuntando a `http://IP_ESP32/capture` si añades ese modo, o
simplemente una foto con el móvil desde el mismo ángulo) y ejecuta:

```bash
python calibrate.py empty foto_tablero_vacio.jpg
```

Después, para cada letra de tu set de fichas, coloca UNA ficha en una casilla
conocida (por ejemplo fila 0, columna 0) y ejecuta:

```bash
python calibrate.py letter foto_A.jpg A 0 0
python calibrate.py letter foto_B.jpg B 0 0
...
```

Repite para todas las letras que tenga tu set: A-Z, Ñ, y si tu edición las
incluye como fichas propias, también CH, LL, RR.

## 5. Arrancar el servidor

```bash
python server.py --players 2
```

Abre `http://TU_PC:5000/` para ver el marcador en vivo. Los turnos se asignan
de forma automática y rotativa (jugador 1, 2, 3... y vuelve a empezar) cada
vez que se detecta una jugada válida y estable.

## Limitaciones importantes (léelo antes de confiar en el marcador)

- **El reconocimiento de letras es el eslabón más débil.** El template
  matching funciona razonablemente si la luz e iluminación se mantienen
  constantes entre la calibración y la partida real, pero puede confundir
  letras parecidas. Revisa el panel web tras cada jugada.
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
- El turno se asigna por rotación automática; si un jugador pasa turno sin
  jugar, tendrás que ajustar `current_player` manualmente en
  `game_state.json` o añadir un botón físico en la ESP32-CAM (GPIO) que
  llame a un endpoint `/skip_turn` (no incluido, fácil de añadir).
