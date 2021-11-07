import os

from torch.utils.data import Dataset

import configs
from src.data_utils import read_lines, process_jsonl2io_parallel


class FeverDataset(Dataset):
    def __init__(self, filepath):
        super(FeverDataset, self).__init__()
        lines = read_lines(filepath)
        self.dataset = process_jsonl2io_parallel(lines)  # (input, class_label) tuple list.

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


dataset = FeverDataset(os.path.join(configs.DATA_DIR, 'fever', 'dev.jsonl'))
print(dataset)
