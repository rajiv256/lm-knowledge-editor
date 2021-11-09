import torch.nn as nn


class KnowledgeEditor(nn.Module):
    def __init__(self):
        super(KnowledgeEditor, self).__init__()

        self.finetuned_bert = None
        self.bilstm = nn.LSTM()
        self.alpha = nn.Linear()
        self.beta = nn.Linear()
        self.gamma = nn.Linear()
        self.delta = nn.Linear()
        self.eta = nn.Linear()
