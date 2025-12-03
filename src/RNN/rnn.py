import torch
from typing import Tuple
import torch.nn.functional as F

INPUT_DIMENSIONS = (1, 2)
HIDDEN_DIMENSIONS = (2, 2)
OUTPUT_DIMENSIONS = (2, 1)

class RNN:
    def __init__(self):
        gen = torch.Generator().manual_seed(42)
        self.InputWeights = torch.randn(INPUT_DIMENSIONS, generator=gen)
        self.HiddenWeights = torch.randn(HIDDEN_DIMENSIONS, generator=gen)
        self.OutputWeights = torch.randn(OUTPUT_DIMENSIONS, generator=gen)
        self.HiddenBias = torch.randn(HIDDEN_DIMENSIONS[1], generator=gen)
        self.OutputBias = torch.randn(OUTPUT_DIMENSIONS[1], generator=gen)
        self.epochs = 5

    def forward_pass(self, X_seq: torch.tensor, Y: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        previous_hidden = None
        hidden = torch.zeros((X_seq.shape[0], self.InputWeights.shape[1]))
        outputs = torch.zeros((X_seq.shape[0], self.OutputWeights.shape[1]))

        for t in range(X_seq.shape[0]):
            # Input layer
            X = X_seq[t].unsqueeze((0))
            X_i = X @ self.InputWeights

            # Hidden layer
            if previous_hidden is None: 
                Xh_t = F.relu(X_i)
            else:
                X_h = previous_hidden @ self.HiddenWeights
                Xh_t = F.tanh(X_i + X_h + self.HiddenBias)
            previous_hidden = Xh_t

            hidden[t, :] = Xh_t
            
            # Output layer
            logit = Xh_t @ self.OutputWeights + self.OutputBias # Y_hat
            outputs[t, :] = logit 
            # print(outputs)
        
        # Return the hidden state for grad descent, output of the RNN 
        return hidden, outputs # was outputs[-1]

    def backward_pass(self, sequence, hidden, dy):
        next_hidden = None
        input_weight_grad, output_weight_grad, hidden_weight_grad, hidden_bias_grad, output_bias_grad = [0] * 5

        for t in range(hidden.shape[0]-1, -1, -1):
            output_bias_grad += dy # dL/dd 
            output_weight_grad += (dy * hidden[t]) # dL/dWy
            
            print(f'W_y grad: {output_weight_grad}')
            print(f'Output bias grad: {output_bias_grad}')
            if next_hidden is not None:
                h_grad = dy * self.OutputWeights.T + next_hidden
            else:
                h_grad = dy * self.OutputWeights.T 
            # print(output_bias_grad.item())
            tanh_grad = 1 - hidden[t]**2 # dh/dtanh
            h_grad = h_grad * tanh_grad
            next_hidden = h_grad 

            if t > 0:
                hidden_weight_grad += h_grad * hidden[t-1]
            input_weight_grad += h_grad * sequence[t]
            hidden_bias_grad += h_grad
            print(f'W_h grad: {hidden_weight_grad}')
            print(f'W_x grad: {input_weight_grad}')
            print(f'Hidden bias grad: {hidden_bias_grad}')
            print("------------------------------------------")

            learning_rate = 1e-6

    def train(self, X: torch.tensor, Y: torch.tensor):
        for _ in range(self.epochs): # e = number of epoch we are at
            for index, sequence in enumerate(X[:self.epochs]):
                hidden, y_pred = self.forward_pass(sequence, Y[index]) # y_pred is all outputs
                y_hat = y_pred[-1]
                # print(f'sequence/index: {sequence}/{index}')
                # print(f'index: {index}, y value: {Y[index]}')
                # print(hidden)
                # print("-")

                loss = 1/2 * ((y_hat - Y[index]) ** 2) # Squared error 
                # print(f'loss: {loss.item()}')

                dy = (y_hat - Y[index]) # dL/dy

                self.backward_pass(sequence, hidden, dy)