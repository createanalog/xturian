# -*- coding: utf-8 -*-
#
# Mainly OpenCV processing here. Made with a lot of IA help and misshelp. Fk. IA.
#
 
import os
import importlib
import cv2
import numpy as np
import logging

from board_config import BOARD_SIZE, BOARD_LAYOUT, LETTER_VALUES, BINGO_BONUS, BINGO_TILE_COUNT

logger = logging.getLogger(__name__)

SHIT_CAMERA = False

WARPED_SIZE = 600          
CELL_SIZE = WARPED_SIZE // BOARD_SIZE

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

_start_board_image = None
_start_hsv_tile_info = None
_last_board_image = None
_last_known_corners = None  # Keep using last known coorners, until able to detect them again.


# OCR Engine handling

_OCR_ENGINE_MODULES = {
    "easyocr": "lrecog_easyocr",
    "paddleocr": "lrecog_paddleocr",
}
DEFAULT_OCR_ENGINE = "paddleocr"

_active_engine = None
_active_engine_name = DEFAULT_OCR_ENGINE


def set_ocr_engine(name):

    global _active_engine, _active_engine_name
    name = name.lower()
    if name not in _OCR_ENGINE_MODULES:
        raise ValueError(
            f"Invalid OCR: {name!r}. Valid options are: "
            f"{list(_OCR_ENGINE_MODULES)}"
        )
    _active_engine = importlib.import_module(_OCR_ENGINE_MODULES[name])
    _active_engine_name = name


def _ocr_engine():
    global _active_engine
    if _active_engine is None:
        set_ocr_engine(_active_engine_name)
    return _active_engine


def get_ocr_reader():
    return _ocr_engine().get_ocr_reader()

#
# Returns center coordinates of aruco markers: top-left, top-right, bottom-right, bottom-left
# Please be aware of aruco dictionary used here.
#
def find_board_corners(image):

    global _last_known_corners
    
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or len(ids) < 4:
        return _last_known_corners

    ids = ids.flatten()
    needed = {0, 1, 2, 3}
    if not needed.issubset(set(ids)):

        return _last_known_corners

    centers = {}
    for i, marker_id in enumerate(ids):
        if marker_id in needed:
            pts = corners[i][0]
            centers[marker_id] = pts.mean(axis=0)

    _last_known_corners = np.array([centers[0], centers[1], centers[2], centers[3]], dtype=np.float32)
    return _last_known_corners


#
# Perspective correction
#
def _warp_board(image, src_points):

    dst_points = np.array([
        [0, 0],
        [WARPED_SIZE - 1, 0],
        [WARPED_SIZE - 1, WARPED_SIZE - 1],
        [0, WARPED_SIZE - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (WARPED_SIZE, WARPED_SIZE))
    
    if SHIT_CAMERA:
        warped = _filterForShitCameras(warped)
   
    return warped


#
# Extract relevant information from image
#
def imageSeen(image):

    global _start_board_image, _start_hsv_tile_info, _last_board_image

    waiting_for_corners = _last_known_corners is None
    
    corners = find_board_corners(image)
    
    if corners is None:
        return None

    warped = _warp_board(image, corners)
    
    if waiting_for_corners:
        logger.info("Corners detected")
        _start_board_image = warped 
  
        #image_20x20_bgr = cv2.resize(warped, (BOARD_SIZE, BOARD_SIZE), interpolation=cv2.INTER_AREA)
        image_20x20_bgr = _medianForEachTile(warped)  # median is a little bit better than avereage 

        #update_tile_view(image_20x20_bgr)
        
        _start_hsv_tile_info = cv2.cvtColor(image_20x20_bgr, cv2.COLOR_BGR2HSV)
          
    _last_board_image = warped

    _findChangedTiles()

#
# Could be use to find changed tiles.
#
def _findChangedTiles() :

    # trial-error thresholds
    #threshold_h = 0.2 * 90.0    # 18
    threshold_h = 0.4 * 90.0     #36   
    #threshold_s = 0.15 * 255.0  # 38.25
    threshold_s = 0.25 * 255.0  # 63
    threshold_v = 0.3 * 255.0   # 76.5
    
    global _start_hsv_tile_info

    image_20x20_bgr = _medianForEachTile(_last_board_image)
    hsv_now = cv2.cvtColor(image_20x20_bgr, cv2.COLOR_BGR2HSV)

    
    hsv1 = _start_hsv_tile_info.astype(np.float32)
    hsv2 = hsv_now.astype(np.float32)
    
    diff = np.abs(hsv1 - hsv2)

    diff[:, :, 0] = np.minimum(diff[:, :, 0], 180.0 - diff[:, :, 0])

    mask_h = diff[:, :, 0] > threshold_h
    mask_s = diff[:, :, 1] > threshold_s
    mask_hs = (diff[:, :, 0] > threshold_h/5) & (diff[:, :, 1] > threshold_s/2)
    #mask_v = diff[:, :, 2] > threshold_v
    mask_exceeded = mask_h | mask_s | mask_hs #| mask_v

    rows, cols, _ = _start_hsv_tile_info.shape
    for y in range(rows):
        for x in range(cols):
            if mask_exceeded[y, x]:
                h1, s1, v1 = _start_hsv_tile_info[y, x]
                h2, s2, v2 = hsv_now[y, x]
                dh = diff[y, x, 0]
                ds = diff[y, x, 1]
                dv = diff[y, x, 2]

                logger.info(f"Desviación > 30% en Posición (y={y}, x={x}):")
                logger.info(f"  - Start   -> H: {h1}, S: {s1}, V:{v1}")
                logger.info(f"  - Current -> H: {h2}, S: {s2}, V:{v2}")
                logger.info(f"  - Diff    -> ΔH: {dh:.1f}, ΔS: {ds:.1f}, ΔV: {dv:.1f}\n")


#
# Calculates tile median color code. A little better than avereage.
#
def _medianForEachTile(img):

    h, w = img.shape[:2]

    image_20x20_bgr = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)

    y_edges = np.linspace(0, h, BOARD_SIZE + 1, dtype=int)
    x_edges = np.linspace(0, w, BOARD_SIZE + 1, dtype=int)

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
 
            patch = img[
                y_edges[i] : y_edges[i + 1], x_edges[j] : x_edges[j + 1]
            ]

            image_20x20_bgr[i, j] = np.median(patch, axis=(0, 1)).astype(np.uint8)

    return image_20x20_bgr


#
# Show hsv tile information for start image. Used just for checking.
#
def update_tile_view(image_20x20_bgr, nombre_ventana="Start board HSV info", escala_pixel=30 ):
    
    h, w, _ = image_20x20_bgr.shape
    ancho_vis = w * escala_pixel
    alto_vis = h * escala_pixel

    vista = cv2.resize(
        image_20x20_bgr,
        (ancho_vis, alto_vis),
        interpolation=cv2.INTER_NEAREST,
    )

    for x in range(0, ancho_vis, escala_pixel):
      cv2.line(vista, (x, 0), (x, alto_vis), (50, 50, 50), 1)
    for y in range(0, alto_vis, escala_pixel):
      cv2.line(vista, (0, y), (ancho_vis, y), (50, 50, 50), 1)

    cv2.imshow(nombre_ventana, vista)
   
    cv2.waitKey(1)

#
# Helps, just a little bit, when using old/low quality cameras/light
#
def _filterForShitCameras(img): 

    # CLAHE to normalize local contrast 

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced_color = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Unsharp masking
    
    blur = cv2.GaussianBlur(enhanced_color, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(enhanced_color, 1.5, blur, -0.5, 0)

    return sharpened
    

def read_board(image):

    if _last_known_corners is None:
        return None
 
    return _ocr_engine().recognize_board(
        _last_board_image, CELL_SIZE, BOARD_SIZE
    )

