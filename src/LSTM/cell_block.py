import torch
from gates import ForgetGate, InputGate, CandidateGate, OutputGate
from activations import Tanh

class Cell:
    def __init__(self, input_dims, hidden_dims):
        # Forget gate
        self.Wf = torch.randn((input_dims + hidden_dims, hidden_dims)) * 0.01
        self.bf = torch.randn((1, hidden_dims))
        self.Ft = ForgetGate()
        # Input gate
        self.Wi = torch.randn((input_dims + hidden_dims, hidden_dims)) * 0.01
        self.bi = torch.randn((1, hidden_dims))
        self.It = InputGate()
        # Candidate gate 
        self.Wc = torch.randn((input_dims + hidden_dims, hidden_dims)) * 0.01
        self.bc = torch.randn((1, hidden_dims))
        self.Ct = CandidateGate()
        # Output gate 
        self.Wo = torch.randn((input_dims + hidden_dims, hidden_dims)) * 0.01
        self.bo = torch.randn((1, hidden_dims))
        self.Ot = OutputGate()

    def forward(self, prev_ct: torch.tensor, prev_h: torch.tensor, x: torch.tensor):
        Xt = torch.cat((x, prev_h), dim=1)
        ft = self.Ft.forward(self.Wf, Xt, self.bf)

        cell_state = prev_ct * ft

        it = self.It.forward(self.Wi, Xt, self.bi)

        ct = self.Ct.forward(self.Wc, Xt, self.bc)

        cell_state += ct * it

        ot = self.Ot.forward(self.Wo, Xt, self.bo)

        t = Tanh() 
        ht = ot * t.forward(cell_state)

        return cell_state, ht
