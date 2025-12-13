import torch 
from activations import Sigmoid, Tanh

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
    def forward(self, Wc: torch.tensor, Xt: torch.tensor,
                 bc: torch.tensor) -> torch.tensor:
        t = Tanh()
        return t.forward(Xt @ Wc + bc)

class OutputGate:
    def forward(self, Wo: torch.tensor, Xt: torch.tensor,
                bo: torch.tensor) -> torch.tensor:
        s = Sigmoid() 
        return s.forward(Xt @ Wo + bo)