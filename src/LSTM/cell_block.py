import torch
from gates import ForgetGate, InputGate, CandidateGate, OutputGate
from activations import Tanh
from typing import Tuple

States = Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, 
               torch.tensor, torch.tensor, torch.tensor, torch.tensor]

class Cell:
    def __init__(self, input_dims, hidden_dims, output_dims):
        gen = torch.Generator().manual_seed(42)
        # Forget gate
        self.Wf = torch.randn((input_dims + hidden_dims, hidden_dims), generator=gen) * 0.1
        self.bf = torch.randn((1, hidden_dims), generator=gen)
        self.Ft = ForgetGate()
        # Input gate
        self.Wi = torch.randn((input_dims + hidden_dims, hidden_dims), generator=gen) * 0.1
        self.bi = torch.randn((1, hidden_dims), generator=gen)
        self.It = InputGate()
        # Candidate gate 
        self.Wc = torch.randn((input_dims + hidden_dims, hidden_dims), generator=gen) * 0.1
        self.bc = torch.randn((1, hidden_dims), generator=gen)
        self.Ct = CandidateGate()
        # Output gate 
        self.Wo = torch.randn((input_dims + hidden_dims, hidden_dims), generator=gen) * 0.1
        self.bo = torch.randn((1, hidden_dims), generator=gen)
        self.Ot = OutputGate()
        
        self.Wy = torch.randn((hidden_dims, output_dims), generator=gen) * 0.1
        self.by = torch.randn((1, output_dims), generator=gen)

    def forward(self, prev_ct: torch.tensor, prev_h: torch.tensor, x: torch.tensor) -> States:
        Xt = torch.cat((x, prev_h), dim=1)
        ft = self.Ft.forward(self.Wf, Xt, self.bf)

        cell_state = prev_ct * ft

        it = self.It.forward(self.Wi, Xt, self.bi)

        ct = self.Ct.forward(self.Wc, Xt, self.bc)

        cell_state += ct * it

        ot = self.Ot.forward(self.Wo, Xt, self.bo)

        t = Tanh() 
        ht = ot * t.forward(cell_state)

        zt = ht @ self.Wy + self.by # of shape (1, 1)

        return Xt, ft, it, ct, ot, cell_state, ht, zt

    def backward(self, Xt: torch.tensor, ot: torch.tensor, cell_state: torch.tensor, 
                 hidden_state: torch.tensor, dZ: torch.tensor, xt: torch.tensor):
        T = hidden_state.shape[1]
        tan = Tanh() 
        dWf, dWi = torch.zeros_like(self.Wf), torch.zeros_like(self.Wi)
        dWc, dWo = torch.zeros_like(self.Wc), torch.zeros_like(self.Wo)
        dWy, dby = torch.zeros_like(self.Wy), torch.zeros_like(self.by)
        dbo, dbc = torch.zeros_like(self.bo), torch.zeros_like(self.bc)
        dbi, dbf = torch.zeros_like(self.bi), torch.zeros_like(self.bf)

        dWy = dZ @ hidden_state # dZ/dWy
        dby = dZ # dZ/dby

        print(f'dWy: {dWy}')
        print(f'dby: {dby}')
        for t in range(T-1, -1, -1):
            real_ot = ot * (1- ot)
            dht = dZ @ self.Wy.T
            
            dWo += dht * tan.forward(cell_state) * real_ot @ Xt # dL/dWo
            print(f'dWo: {dWo}')