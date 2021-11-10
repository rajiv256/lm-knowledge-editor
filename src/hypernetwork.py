import torch.nn as nn
import torch.nn.functional as F


class KnowledgeEditor(nn.Module):
    def __init__(self,
                 n,
                 m,
                 input_size=768,  # verified
                 hidden_size=128,  # verified
                 linear_out=1024,  # verified
                 num_layers=1):  # verified
        """
        Args:
            n:
            m:
            input_size:
            hidden_size:
            linear_out:
            num_layers:
        """
        super(KnowledgeEditor, self).__init__()

        self.bilstm = nn.LSTM(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bidirectional=True)
        self.linear = nn.Linear(2*hidden_size, linear_out)
        self.alpha_linear = nn.Linear(hidden_size, m)
        self.beta_linear = nn.Linear(hidden_size, m)
        self.gamma_linear = nn.Linear(hidden_size, n)
        self.delta_linear = nn.Linear(hidden_size, n)

        # TODO(rajiv): Maybe an intermediate layer?
        self.eta_linear = nn.Linear(hidden_size, 1)

    def forward(self, X, gradW):
        """

        Args:
            X: Vector(input [SEP] SUPPORTS [SEP] REFUTES)
            gradW: gradients of `finetuned_bert`.
        Returns:

        """

        # TODO(rajiv): Make sure X is input + y +  a and not just input.
        bilstm_h, state = self.bilstm(X)
        output = F.tanh(self.linear(bilstm_h))
        alpha = self.alpha_linear(output)
        beta = self.beta_linear(output)
        gamma = self.gamma_linear(output)
        delta = self.delta_linear(output)
        eta = self.eta_linear(output)

        # TODO(rajiv): While computing *_hat, we are assuming that the first
        # dimension would be the batch dimension. So we transpose the last two
        # dimensions.
        alpha_hat = F.softmax(alpha) * gamma.unsqueeze(-1).transpose(-2, -1)
        beta_hat = F.softmax(beta) * delta.unsqueeze(-1).transpose(-2, -1)

        delW = F.sigmoid(eta) * (F.dot(alpha_hat, gradW) + beta_hat)
