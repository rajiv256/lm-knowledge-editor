import os

import requests

from configs import DATA_DIR

FEVER_DATA_URLS = {
    'fever_train': "http://dl.fbaipublicfiles.com/KILT/fever-train-kilt.jsonl",
    'fever_dev': "http://dl.fbaipublicfiles.com/KILT/fever-dev-kilt.jsonl",
    'fever_test': "http://dl.fbaipublicfiles.com/KILT/fever-test_without_answers-kilt.jsonl"
}

if __name__ == "__main__":
    for file, url in FEVER_DATA_URLS.items():
        req = requests.get(url, allow_redirects=True)
        with open(os.path.join(DATA_DIR, file + '.jsonl'), 'wb') as fout:
            fout.write(req.content)
