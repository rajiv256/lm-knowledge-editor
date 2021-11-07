import json
import logging
from concurrent.futures import ThreadPoolExecutor

from configs import SPECIAL_CHARS, LABEL_MAP, SPECIAL_NUMS


def read_lines(filepath):
    lines = []
    with open(filepath, 'r') as fin:
        lines = fin.readlines()
    lines = [line.strip() for line in lines]
    logging.info(f'Read {len(lines)} lines from file: {filepath}')
    return lines


def fever_jsonl2io(line):
    d = json.loads(line)
    input_sentence = d['input']
    try:
        output = d['output']
        answer = output[0]['answer']  # SUPPORTS or REFUTES
        label = LABEL_MAP[answer]
    except:
        label = SPECIAL_NUMS["NEG_ONE"]
    return input_sentence, label


def process_jsonl2io_parallel(lines):
    dataset = []
    with ThreadPoolExecutor() as executor:
        for (input_sentence, label) in executor.map(fever_jsonl2io, lines):
            if label is not SPECIAL_NUMS["NEG_ONE"]:
                dataset.append((input_sentence, label))
    return dataset


def write_lines(lines, filepath):
    with open(filepath, 'w') as fout:
        all_lines = SPECIAL_CHARS["NEWLINE"].join(lines)
        fout.write(all_lines)
        logging.info(f'Wrote {len(all_lines)} to filepath: {filepath}')
