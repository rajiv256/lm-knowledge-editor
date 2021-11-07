import os

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LABEL_MAP = {
    'SUPPORTS': 1,
    'REFUTES': 0
}
SPECIAL_CHARS = {
    "NEWLINE": "\n",
    "SPACE": " ",
    "TAB": "\t"
}
SPECIAL_NUMS = {
    "NEG_ONE": -1,
    "ZERO": 0,
    "ONE": 1
}
