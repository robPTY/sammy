import torch
from typing import Tuple, List
import torch.nn.functional as F

Gradients = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

class RNN:
    def __init__(self):
        gen = torch.Generator().manual_seed(42)
        self.INPUT_DIMENSIONS = (1, 2)
        self.HIDDEN_DIMENSIONS = (2, 2)
        self.OUTPUT_DIMENSIONS = (2, 1)
        self.InputWeights = torch.randn(self.INPUT_DIMENSIONS, generator=gen)
        self.HiddenWeights = torch.randn(self.HIDDEN_DIMENSIONS, generator=gen)
        self.OutputWeights = torch.randn(self.OUTPUT_DIMENSIONS, generator=gen)
        self.HiddenBias = torch.randn((1, self.HIDDEN_DIMENSIONS[1]), generator=gen)
        self.OutputBias = torch.randn((1, self.OUTPUT_DIMENSIONS[1]), generator=gen)
        self.epochs = 200
        self.learning_rate = 1e-3

    def forward_pass(self, X_seq: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        previous_hidden = None
        hidden = torch.zeros((X_seq.shape[0], self.InputWeights.shape[1]))
        outputs = torch.zeros((X_seq.shape[0], self.OutputWeights.shape[1]))

        for t in range(X_seq.shape[0]):
            # Input layer
            X = X_seq[t].unsqueeze((0))
            X_i = X @ self.InputWeights

            # Hidden layer
            if previous_hidden is None: 
                Xh_t = X_i
            else:
                X_h = previous_hidden @ self.HiddenWeights
                Xh_t = F.tanh(X_i + X_h + self.HiddenBias)
            previous_hidden = Xh_t

            hidden[t, :] = Xh_t
            
            # Output layer
            logit = Xh_t @ self.OutputWeights + self.OutputBias 
            outputs[t, :] = logit 
        
        # Return the hidden state for grad descent, output of the RNN 
        return hidden, outputs

    def predict(self, x: List[float]) -> float:
        x = torch.tensor(x)
        _, outputs = self.forward_pass(x)
        y_hat = outputs[-1]
        return y_hat.item()

    def backward_pass(self, sequence: torch.tensor, hidden: torch.tensor, dy: torch.tensor) -> Gradients:
        next_hidden = None
        dWx, dWy, dWh, dG, dB = [0] * 5

        for t in range(hidden.shape[0]-1, -1, -1):
            hidden_t = hidden[t, :].unsqueeze((0))

            dB += dy # dL/dB
            dWy += (hidden_t.T @ dy) # dL/dWy
            
            if next_hidden is None:
                h_grad = dy @ self.OutputWeights.T # dy/dh(t)
            else:
                h_grad = dy @ self.OutputWeights.T + next_hidden @ self.HiddenWeights.T 
          
            tanh_grad = 1 - hidden_t**2 #dtanh/dParams

            h_grad = h_grad * tanh_grad # dh/dtanh 
            next_hidden = h_grad 
            
            previous_hidden = hidden[t-1].unsqueeze((0))
            if t > 0:
                dWh += previous_hidden.T @ h_grad # dL/dWh
                dG += h_grad # dL/dG

            dWx += h_grad * sequence[t] # dL/dWx

        return dWy, dB, dWh, dWx, dG
    
    def sgd_step(self, dWy: torch.tensor, dB: torch.tensor, dWh: torch.tensor, 
                 dWx: torch.tensor, dG: torch.tensor, learning_rate: float)-> None:
        self.InputWeights -= (learning_rate * dWx)
        self.OutputWeights -= (learning_rate * dWy)
        self.HiddenWeights -= (learning_rate * dWh)
        self.HiddenBias -= (learning_rate * dG)
        self.OutputBias -= (learning_rate * dB)
    
    def calculate_loss(self, X: torch.tensor, Y: torch.tensor) -> float:
        _, outputs = self.forward_pass(X)
        y_pred = outputs[-1]
        loss = 1/2 * (y_pred - Y)**2
        return loss.item()

    def calculate_total_loss(self, X: torch.tensor, Y: torch.tensor) -> float:
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / float(len(Y))

    def train(self, X: torch.tensor, Y: torch.tensor, X_valid: torch.tensor,
              Y_valid: torch.tensor) -> List[float]:
        losses = []
        for e in range(self.epochs): # e = number of epoch we are at
            epoch_loss = 0
            for index, sequence in enumerate(X):
                hidden, y_pred = self.forward_pass(sequence) # y_pred is all outputs
                y_hat = y_pred[-1]

                loss = 1/2 * ((y_hat - Y[index]) ** 2) # Squared error 
                dy = (y_hat - Y[index]).view(1, 1) # dL/dy
                epoch_loss += loss.item()

                dWy, dB, dWh, dWx, dG = self.backward_pass(sequence, hidden, dy)
                self.sgd_step(dWy, dB, dWh, dWx, dG, self.learning_rate)
            
            valid_loss = 0
            for index, sequence in enumerate(X_valid):
                _, outputs = self.forward_pass(sequence)
                loss = 1/2 * ((outputs[-1] - Y_valid[index]) **2)
                valid_loss += loss.item()

            print(f"epoch {e+1}, train loss = {epoch_loss/len(X)}, valid loss = {valid_loss/len(X_valid)}")
        return losses