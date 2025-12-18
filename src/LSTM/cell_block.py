import torch
from gates import ForgetGate, InputGate, CandidateGate, OutputGate
from activations import Tanh
from typing import Tuple, List, Dict

States = Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, 
               torch.tensor, torch.tensor, torch.tensor, torch.tensor]

class Cell:
    def __init__(self, input_dims, hidden_dims, output_dims):
        gen = torch.Generator().manual_seed(42)
        # Forget gate
        self.Wf = torch.randn((input_dims + hidden_dims, hidden_dims), generator=gen) * 0.1
        self.bf = torch.randn((1, hidden_dims), generator=gen)
        self.Ft = ForgetGate()
        self.dWf = torch.zeros_like(self.Wf)
        self.dbf = torch.zeros_like(self.bf)
        # Input gate
        self.Wi = torch.randn((input_dims + hidden_dims, hidden_dims), generator=gen) * 0.1
        self.bi = torch.randn((1, hidden_dims), generator=gen)
        self.It = InputGate()
        self.dWi = torch.zeros_like(self.Wi)
        self.dbi = torch.zeros_like(self.bi)
        # Candidate gate 
        self.Wc = torch.randn((input_dims + hidden_dims, hidden_dims), generator=gen) * 0.1
        self.bc = torch.randn((1, hidden_dims), generator=gen)
        self.Ct = CandidateGate()
        self.dWc = torch.zeros_like(self.Wc)
        self.dbc = torch.zeros_like(self.bc)
        # Output gate 
        self.Wo = torch.randn((input_dims + hidden_dims, hidden_dims), generator=gen) * 0.1
        self.bo = torch.randn((1, hidden_dims), generator=gen)
        self.Ot = OutputGate()
        self.dWo = torch.zeros_like(self.Wo)
        self.dbo = torch.zeros_like(self.bo)
        # Network output
        self.Wy = torch.randn((hidden_dims, output_dims), generator=gen) * 0.1
        self.by = torch.randn((1, output_dims), generator=gen)
        self.dWy, self.dby = torch.zeros_like(self.Wy), torch.zeros_like(self.by)

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

    def backward(self, states_cache: List[Dict], dZ: torch.tensor):
        # The cache is being accesed at time step t
        Xt = states_cache["Xt"]
        ft, it = states_cache["ft"], states_cache["it"]
        ot, ct = states_cache["ot"], states_cache["candidate"]
        prev_cst, cell_state = states_cache["prev_cst"], states_cache["cell_state"]
        hidden_state = states_cache["prev_hidden"]
        tan = Tanh() 

        dZ = dZ.view(1, -1)  
        self.dWy += hidden_state.T @ dZ # dZ/dWy
        self.dby += dZ # dZ/dby

        real_ot = ot * (1 - ot) # dOt/dSigm
        dht = dZ @ self.Wy.T # dz/dh(t)
        dot = tan.forward(cell_state) # dh/dOt
            
        self.dWo += Xt.T @ dht * dot * real_ot  # dL/dWo
        self.dbo += dht * dot * real_ot # dL/dbo