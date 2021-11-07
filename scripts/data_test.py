import json
import os

from configs import DATA_DIR


def test_0():
    filepath = os.path.join(DATA_DIR, 'dev.jsonl')
    with open(filepath, 'r') as fin:
        lines = fin.readlines()
    line = lines[0]
    data = json.loads(line)
    print(data)


if __name__ == "__main__":
    test_0()
