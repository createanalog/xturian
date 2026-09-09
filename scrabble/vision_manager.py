# -*- coding: utf-8 -*-
#
# Mainly OpenCV processing here. Made with a lot of IA help and misshelp. Fk. IA.
#
 
import os
import importlib
import cv2
import numpy as np

from board_config import BOARD_SIZE, BOARD_LAYOUT, LETTER_VALUES, BINGO_BONUS, BINGO_TILE_COUNT

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
    return _active_engine.get_ocr_reader()

TEMPLATES_DIR = "board_templates"
EMPTY_BOARD_PATH = os.path.join(TEMPLATES_DIR, "empty_board.jpg")

WARPED_SIZE = 600           # after perspective correction
CELL_SIZE = WARPED_SIZE // BOARD_SIZE

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()


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


# Keep using last known coorners, until able to detect them again.
_last_known_corners = None


def warp_board(image, src_points):
    dst_points = np.array([
        [0, 0],
        [WARPED_SIZE - 1, 0],
        [WARPED_SIZE - 1, WARPED_SIZE - 1],
        [0, WARPED_SIZE - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (WARPED_SIZE, WARPED_SIZE))
    return warped


def read_board(image):

    corners = find_board_corners(image)
    print("CORNERS")
    if corners is None:
        return None
    print("DETECTING")   
  
    warped = warp_board(image, corners)
 
    return _ocr_engine().recognize_board(
        warped, CELL_SIZE, BOARD_SIZE
    )
