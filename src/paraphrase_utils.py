"""
#TODO(rajiv/greg): Don't forget installing dependencies beforehand.

```
pip install bitarray fastBPE hydra-core omegaconf regex requests sacremoses subword_nmt
```

This code is mostly taken from this tutorial:
https://pytorch.org/hub/pytorch_fairseq_translation/
"""
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load an En-Fr Transformer model trained on WMT'14 data :
en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')

# Use the GPU (optional):
en2fr.to(device)

# Translate with beam search:
fr = en2fr.translate('Hello world!', beam=5)
# assert fr == 'Bonjour à tous !'
print(fr)

# Roundtrip translation for paraphrase generation
# Round-trip translations between English and German:
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

paraphrase = de2en.translate(en2de.translate('PyTorch Hub is an awesome interface!'))
assert paraphrase=='PyTorch Hub is a fantastic interface!'

# Compare the results with English-Russian round-trip translation:
en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')

paraphrase = ru2en.translate(en2ru.translate('PyTorch Hub is an awesome interface!'))
assert paraphrase=='PyTorch is a great interface!'
