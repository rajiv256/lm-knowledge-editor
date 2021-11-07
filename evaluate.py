import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Loads the model
tokenizer = AutoTokenizer.from_pretrained("nielsr/coref-bert-base")
model = AutoModel.from_pretrained("nielsr/coref-bert-base")

# Load FEVER dataset
fever_dataset = load_dataset("fever", "v1.0")
paper_dev = fever_dataset['paper_dev']
paper_test = fever_dataset['paper_test']

print(type(paper_dev))