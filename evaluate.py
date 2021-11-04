import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Loads the model
tokenizer = AutoTokenizer.from_pretrained("nielsr/coref-bert-base")
model = AutoModel.from_pretrained("nielsr/coref-bert-base")

# Load FEVER dataset
fever_dataset = load_dataset("fever", "wiki_pages")

print(fever_dataset['wikipedia_pages'][1])
