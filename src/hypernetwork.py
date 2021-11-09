import torch.nn as nn
import torch.nn.functional as F


class KnowledgeEditor(nn.Module):
    def __init__(self, n, m, input_size, hidden_size, num_layers):
        super(KnowledgeEditor, self).__init__()

        self.bilstm = nn.LSTM(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bidirectional=True)
        self.alpha_linear = nn.Linear(hidden_size, m)
        self.beta_linear = nn.Linear(hidden_size, m)
        self.gamma_linear = nn.Linear(hidden_size, n)
        self.delta_linear = nn.Linear(hidden_size, n)

        # TODO(rajiv): Maybe an intermediate layer?
        self.eta_linear = nn.Linear(hidden_size, 1)


    def forward(self, x, gradW):
        """

        Args:
            x: input sentence vector
            gradW: gradients of `finetuned_bert`.
        Returns:

        """
        bilstm_h, state = self.bilstm(x)
        alpha = self.alpha_linear(bilstm_h)
        beta = self.beta_linear(bilstm_h)
        gamma = self.gamma_linear(bilstm_h)
        delta = self.delta_linear(bilstm_h)
        eta = self.eta_linear(bilstm_h)

        # TODO(rajiv): While computing *_hat, we are assuming that the first
        # dimension would be the batch dimension. So we transpose the last two
        # dimensions.
        alpha_hat = F.softmax(alpha)*gamma.unsqueeze(-1).transpose(-2, -1)
        beta_hat = F.softmax(beta)*delta.unsqueeze(-1).transpose(-2, -1)

        delW = F.sigmoid(eta)*(F.dot(alpha_hat, gradW) + beta_hat)