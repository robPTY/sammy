import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoRNN(nn.Module):
    def __init__(self, manual_rnn):
        super().__init__()
        self.Wx = nn.Parameter(manual_rnn.InputWeights.clone().detach())
        self.Wh = nn.Parameter(manual_rnn.HiddenWeights.clone().detach())
        self.Wy = nn.Parameter(manual_rnn.OutputWeights.clone().detach())
        self.bh = nn.Parameter(manual_rnn.HiddenBias.clone().detach())
        self.by = nn.Parameter(manual_rnn.OutputBias.clone().detach())

    def forward(self, X_seq):
        T = X_seq.shape[0]

        previous_hidden = None
        
        for t in range(T):
            x_t = X_seq[t].unsqueeze(0)

            X_i = x_t @ self.Wx   

            if previous_hidden is None:
                Xh_t = X_i
            else:
                X_h = previous_hidden @ self.Wh
                Xh_t = torch.tanh(X_i + X_h + self.bh)

            previous_hidden = Xh_t

        logit = previous_hidden @ self.Wy + self.by
        return logit
