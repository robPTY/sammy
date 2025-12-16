import torch

class Sigmoid:
    def forward(self, x: torch.tensor) -> torch.tensor:
        return 1.0 / (1.0 + torch.exp(-x))
    
class Tanh:
    def forward(self, x: torch.tensor) -> torch.tensor:
        return (torch.exp(x) - torch.exp(-x))/ (torch.exp(x) + torch.exp(-x))

    def backward(self, x):
        pass 
    
class h:
    def forward(self, x: torch.tensor) -> torch.tensor:
        return (2.0 / (1.0 + torch.exp(-x))) - 1.0

class g:
    def forward(self, x: torch.tensor) -> torch.tensor:
        return (4.0 / (1.0 + torch.exp(-x))) - 2.0