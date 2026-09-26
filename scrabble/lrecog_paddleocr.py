import cv2

MULTIPLIER_DIGITS = "123"

DIGIT_TO_LETTER_CORRECTIONS = {
    "5": "S",
    "1": "I",
    "0": "O"
}

VALID_LETTERS = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"

OCR_MIN_CONFIDENCE = 0.1 # tried 0.3 but left out some letters sometimes

_ocr = None


def get_ocr_reader():

    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
      
        _ocr = PaddleOCR(
            lang="es",
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
            cpu_threads=1,
        )
    return _ocr


def _text_to_cells(text, bbox, cell_size, board_size, grid):

    text = text.upper()

    x_min = float(bbox[:, 0].min())
    x_max = float(bbox[:, 0].max())
    y_center = float(bbox[:, 1].mean())

    char_w = (x_max - x_min) / len(text)

    for i, ch in enumerate(text):
        cx = x_min + char_w / 2 + char_w * i
        col = int(cx // cell_size)
        row = int(y_center // cell_size)

        ch = DIGIT_TO_LETTER_CORRECTIONS.get(ch, ch)
        
        if ch not in VALID_LETTERS:
            continue 

        grid[row][col] = ch


def recognize_board(image, cell_size, board_size):

    grid = [[None] * board_size for _ in range(board_size)]
  
    reader = get_ocr_reader()

    try:
        results = reader.predict(image)
    except Exception as e:
        print(f"[lrecog_paddleocr] PaddleOCR error: {e}")
        return grid
    
    if results is not None: 
       
        for res in results:
            textos = res.get("rec_texts", [])
            cajas = res.get("dt_polys", [])
            puntajes = res.get("rec_scores", [1.0] * len(textos))

            for texto, bbox, score in zip(textos, cajas, puntajes):
                if not texto or score < OCR_MIN_CONFIDENCE:
                    continue
                print(f"Text found: {texto}")
                _text_to_cells(
                    texto, bbox, cell_size, board_size, grid,
                )
    else:
        print("[lrecog_paddleocr] No text detected")

    return grid
