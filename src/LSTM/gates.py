import torch 
from activations import Sigmoid

class ForgetGate:
    def forward(self, Wf: torch.tensor, Xt: torch.tensor,
                bf: torch.tensor) -> torch.tensor:
        return Sigmoid.forward(Xt @ Wf.T + bf)

class InputGate:
    pass 

class OutputGate:
    pass 