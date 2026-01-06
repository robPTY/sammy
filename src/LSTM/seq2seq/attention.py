import torch
from typing import Tuple

Attention = Tuple[torch.tensor, torch.tensor, torch.tensor]

class BahdanauAttention:
    def __init__(self, hidden_dim: int, attention_dim: int):
        self.V = torch.randn(attention_dim, 1) * 0.1
        self.W = torch.randn(hidden_dim, attention_dim) * 0.1
        self.U = torch.randn(hidden_dim, attention_dim) * 0.1

        self.dV = torch.zeros_like(self.V)
        self.dW = torch.zeros_like(self.W)
        self.dU = torch.zeros_like(self.U)
    
    def score(self, decoder_hidden: torch.tensor, encoder_hidden: torch.tensor) -> Attention:
        activated = torch.tanh(decoder_hidden @ self.W + encoder_hidden @ self.U)
        scores = activated @ self.V
        alphas = torch.softmax(scores, dim=0)
        context_vector = torch.sum(alphas * encoder_hidden, dim=0, keepdim=True)
        return context_vector, alphas, activated

    def backward(self, d_context: torch.tensor, encoder_hidden: torch.tensor, 
                decoder_hidden: torch.tensor, alphas: torch.tensor, 
                activated: torch.tensor) -> torch.tensor:
        dAlpha = encoder_hidden @ d_context.T # dL/dAlpha (T, hidden) x (hidden, 1) -> (T, 1)
        weighted_sum = (alphas * dAlpha).sum() 
        inner = dAlpha - weighted_sum
        dEnergy = alphas * inner # dL/dEnergy
        self.dV += activated.T @ dEnergy # dL/dV 

        dtanh = 1 - activated ** 2 # dL/dTanh
        dActivated = dEnergy @ self.V.T # dL/dActivated
        dCombined = dActivated * dtanh
        d_decoder_projection = dCombined.sum(dim=0, keepdim=True)
        self.dW += decoder_hidden.T @ d_decoder_projection # dL/dW 
        self.dU += encoder_hidden.T @ dCombined # dL/dU

        dDecoderHidden = d_decoder_projection @ self.W.T
        return dDecoderHidden

    def zero_grad(self):
        self.dV.zero_()
        self.dW.zero_()
        self.dU.zero_()
    
    def sgd_step(self, learning_rate: float):
        self.V -= learning_rate * self.dV
        self.W -= learning_rate * self.dW
        self.U -= learning_rate * self.dU