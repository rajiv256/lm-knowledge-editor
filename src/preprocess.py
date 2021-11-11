import os

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import configs
from src.data_utils import read_lines, process_jsonl2io_parallel


class FeverDatasetFinetune(Dataset):
    def __init__(self,
                 filepath,
                 max_length=512,
                 padding='max_length',
                 truncation=True):
        super(FeverDatasetFinetune, self).__init__()
        lines = read_lines(filepath)

        # Tuple of input, class label
        self.dataset = process_jsonl2io_parallel(lines)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input_sentence = self.dataset[index][0]
        class_label = self.dataset[index][1]
        encoded_input = self.tokenizer(text=input_sentence,
                                       max_length=self.max_length,
                                       padding=self.padding,
                                       truncation=self.truncation,
                                       return_tensors="pt"
                                       )
        return encoded_input, class_label


class FeverDatasetEditor(Dataset):
    def __init__(self,
                 filepath,
                 max_length=512,
                 padding='max_length',
                 truncation=True):
        super(FeverDatasetEditor, self).__init__()
        lines = read_lines(filepath)



dataset = FeverDatasetFinetune(os.path.join(configs.DATA_DIR, 'fever',
                                         'dev.jsonl'))
print(dataset)
dev_loader = DataLoader(dataset, batch_size=1, shuffle=True)
