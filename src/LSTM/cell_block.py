import torch
from gates import ForgetGate

class Cell:
    def __init__(self, hidden_dims):
        # Forget gate
        self.Wf = torch.randn((hidden_dims, hidden_dims))
        self.bf = torch.randn((1, 1))

    def forward(self, prev_h: torch.tensor, x: torch.tensor):
        Xt = torch.cat((x, prev_h), dim=1)
        Ft = ForgetGate.forward(self.Wf, Xt, self.bf)