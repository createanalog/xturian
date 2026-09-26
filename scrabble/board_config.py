# -*- coding: utf-8 -*-
"""
Configuración del tablero de Scrabble en español.

Incluye:
  - Layout estándar 15x15 de casillas premium (igual en todos los idiomas).
  - Valores de letra de la edición clásica en español (incluye CH, LL, RR, Ñ
    como fichas propias). Si tu edición NO tiene esas fichas dobles, ajusta
    LETTER_VALUES y quita las entradas correspondientes.
"""

BOARD_SIZE = 15

# '.'  = normal
# 'DL' = letra doble | 'TL' = letra triple
# 'DW' = palabra doble | 'TW' = palabra triple
# La casilla central (7,7) es DW (la estrella).
_ROW_TW_DL   = ["TW", ".", ".", "DL", ".", ".", ".", "TW", ".", ".", ".", "DL", ".", ".", "TW"]
_ROW_DW_TL_1 = [".", "DW", ".", ".", ".", "TL", ".", ".", ".", "TL", ".", ".", ".", "DW", "."]
_ROW_DW_DL_1 = [".", ".", "DW", ".", ".", ".", "DL", ".", "DL", ".", ".", ".", "DW", ".", "."]
_ROW_DL_DW_1 = ["DL", ".", ".", "DW", ".", ".", ".", "DL", ".", ".", ".", "DW", ".", ".", "DL"]
_ROW_DW_MID  = [".", ".", ".", ".", "DW", ".", ".", ".", ".", ".", "DW", ".", ".", ".", "."]
_ROW_TL      = [".", "TL", ".", ".", ".", "TL", ".", ".", ".", "TL", ".", ".", ".", "TL", "."]
_ROW_DL_MID  = [".", ".", "DL", ".", ".", ".", "DL", ".", "DL", ".", ".", ".", "DL", ".", "."]
_ROW_CENTER  = ["TW", ".", ".", "DL", ".", ".", ".", "DW", ".", ".", ".", "DL", ".", ".", "TW"]

BOARD_LAYOUT = [
    _ROW_TW_DL,
    _ROW_DW_TL_1,
    _ROW_DW_DL_1,
    _ROW_DL_DW_1,
    _ROW_DW_MID,
    _ROW_TL,
    _ROW_DL_MID,
    _ROW_CENTER,
    _ROW_DL_MID,
    _ROW_TL,
    _ROW_DW_MID,
    _ROW_DL_DW_1,
    _ROW_DW_DL_1,
    _ROW_DW_TL_1,
    _ROW_TW_DL,
]

assert len(BOARD_LAYOUT) == BOARD_SIZE
assert all(len(r) == BOARD_SIZE for r in BOARD_LAYOUT)

# Valores de ficha - edición clásica en español (con CH, LL, RR, Ñ)
# Ajusta si tu set no tiene esas fichas especiales.
LETTER_VALUES = {
    "A": 1, "E": 1, "O": 1, "I": 1, "S": 1, "N": 1, "L": 1, "R": 1, "U": 1, "T": 1,
    "D": 2, "G": 2,
    "C": 3, "B": 3, "M": 3, "P": 3,
    "H": 4, "F": 4, "V": 4, "Y": 4, "W": 4,
    "Q": 5,
    "J": 8, "Ñ": 8, "X": 8,
    "Z": 10,
    "#": 0,  # ficha comodín (blanco)
}

# Bonus por usar las 7 fichas del atril en una sola jugada ("bingo")
BINGO_BONUS = 50
BINGO_TILE_COUNT = 7
