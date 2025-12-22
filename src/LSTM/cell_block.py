import torch
from gates import ForgetGate, InputGate, CandidateGate, OutputGate
from activations import Tanh
from typing import Tuple, List, Dict

States = Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, 
               torch.tensor, torch.tensor, torch.tensor, torch.tensor,
               torch.tensor]

class Cell:
    def __init__(self, input_dims, hidden_dims, output_dims):
        self.input_dims = input_dims
        gen = torch.Generator().manual_seed(42)
        # Forget gate
        self.Wf = torch.randn((input_dims + hidden_dims, hidden_dims), generator=gen) * 0.1
        self.bf = torch.ones((1, hidden_dims))
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
        tanh_cell_state = t.forward(cell_state)
        ht = ot * tanh_cell_state

        zt = ht @ self.Wy + self.by # of shape (1, 1)

        return Xt, ft, it, ct, ot, tanh_cell_state, cell_state, ht, zt

    def backward(self, states_cache: List[Dict], dZ: torch.tensor, 
                 dht_next: torch.tensor, dct_next: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        # The cache is being accesed at time step t
        Xt = states_cache["Xt"]
        ft, it = states_cache["ft"], states_cache["it"]
        ot, ct = states_cache["ot"], states_cache["candidate"]
        prev_cst, cell_state = states_cache["prev_cst"], states_cache["cell_state"]
        tanh_cell_state = states_cache["tanh_cell_state"]
        prev_hidden, hidden_state = states_cache["prev_hidden"], states_cache["ht"]
        tan = Tanh() 

        dZ = dZ.view(1, -1)  
        self.dWy += hidden_state.T @ dZ # dZ/dWy
        self.dby += dZ # dZ/dby

        real_ot = ot * (1 - ot) # dOt/dSigm
        dht = dZ @ self.Wy.T + dht_next # dz/dh(t) + gradient from next timestep
            
        self.dWo += Xt.T @ (dht * tanh_cell_state * real_ot)  # dL/dWo
        self.dbo += dht * tanh_cell_state * real_ot # dL/dbo

        dcell_state = dht * ot * (1 - tanh_cell_state**2) + dct_next
        dftdsig = ft * (1 - ft) 

        self.dWf += Xt.T @ (dcell_state * prev_cst * dftdsig) #dL/dWf
        self.dbf += dcell_state * prev_cst * dftdsig # dL/dbf
        
        ditdsig = it * (1 - it)
        self.dWi += Xt.T @ (dcell_state * ct * ditdsig) # dL/dWi
        self.dbi += dcell_state * ct * ditdsig # dL/dbi

        dctdtanh = 1 - ct ** 2
        self.dWc += Xt.T @ (dcell_state * it * dctdtanh) # dL/dWc
        self.dbc += dcell_state * it * dctdtanh # dL/dbc

        dprev_cell_state = dcell_state * ft
        dXt = ((dht * tanh_cell_state * real_ot) @ self.Wo.T + (dcell_state * prev_cst * dftdsig) @ self.Wf.T +
            (dcell_state * ct * ditdsig) @ self.Wi.T + (dcell_state * it * dctdtanh) @ self.Wc.T)

        dx = dXt[:, :self.input_dims] # dL/dX (input)
        dprev_hidden = dXt[:, self.input_dims:] # dL/dH(t-1)

        return dx, dprev_hidden, dprev_cell_state

    def zero_grad(self):
        self.dWy.zero_()
        self.dby.zero_()
        self.dWo.zero_()
        self.dbo.zero_()
        self.dWf.zero_()
        self.dbf.zero_()
        self.dWi.zero_()
        self.dbi.zero_()
        self.dWc.zero_()
        self.dbc.zero_()
    
    def sgd_step(self, learning_rate: float) -> None:
        self.Wy -= learning_rate * self.dWy
        self.by -= learning_rate * self.dby
        self.Wo -= learning_rate * self.dWo
        self.bo -= learning_rate * self.dbo
        self.Wf -= learning_rate * self.dWf
        self.bf -= learning_rate * self.dbf
        self.Wi -= learning_rate * self.dWi 
        self.bi -= learning_rate * self.dbi
        self.Wc -= learning_rate * self.dWc
        self.bc -= learning_rate * self.dbc

    def get_state(self) -> Dict[str, torch.Tensor]:
        return {
            "Wf": self.Wf, "bf": self.bf,
            "Wi": self.Wi, "bi": self.bi,
            "Wc": self.Wc, "bc": self.bc,
            "Wo": self.Wo, "bo": self.bo,
            "Wy": self.Wy, "by": self.by,
        }

    def load_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.Wf = state["Wf"]
        self.bf = state["bf"]
        self.Wi = state["Wi"]
        self.bi = state["bi"]
        self.Wc = state["Wc"]
        self.bc = state["bc"]
        self.Wo = state["Wo"]
        self.bo = state["bo"]
        self.Wy = state["Wy"]
        self.by = state["by"]