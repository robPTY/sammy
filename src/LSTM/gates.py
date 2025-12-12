import torch 
from activations import Sigmoid

class ForgetGate:
    def forward(self, Wf: torch.tensor, Xt: torch.tensor,
                bf: torch.tensor) -> torch.tensor:
        s = Sigmoid()
        return s.forward(Xt @ Wf + bf)

class InputGate:
    def forward(self, Wi: torch.tensor, Xt: torch.tensor,
                bi: torch.tensor) -> torch.tensor:
        s = Sigmoid() 
        return s.forward(Xt @ Wi + bi)

class CandidateGate:
    pass

class OutputGate:
    pass 