import torch
import pandas as pd
from typing import Tuple
import torch.nn.functional as F

INPUT_DIMENSIONS = (1, 2)
HIDDEN_DIMENSIONS = (2, 2)
OUTPUT_DIMENSIONS = (2, 1)
TIME_STEPS = 3

class RNN:
    def __init__(self):
        gen = torch.Generator().manual_seed(42)
        self.InputWeights = torch.randn(INPUT_DIMENSIONS, generator=gen)
        self.HiddenWeights = torch.randn(HIDDEN_DIMENSIONS, generator=gen)
        self.OutputWeights = torch.randn(OUTPUT_DIMENSIONS, generator=gen)
        self.HiddenBias = torch.randn(HIDDEN_DIMENSIONS[1], generator=gen)
        self.OutputBias = torch.randn(OUTPUT_DIMENSIONS[1], generator=gen)
        self.epochs = 100

    def forward_pass(self, X_seq: torch.tensor, Y: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        previous_step = None
        hidden = torch.zeros((X_seq.shape[0], self.InputWeights.shape[1]))
        outputs = torch.zeros((X_seq.shape[0], self.OutputWeights.shape[1]))

        for t in range(X_seq.shape[0]):
            # Input layer
            X = X_seq[t].unsqueeze((0))
            X_i = X @ self.InputWeights

            # Hidden layer
            if previous_step is None: 
                Xh_t = F.relu(X_i)
            else:
                X_h = previous_step @ self.HiddenWeights
                Xh_t = F.tanh(X_i + X_h + self.HiddenBias)
            previous_step = Xh_t

            hidden[t, :] = Xh_t
            
            # Output layer
            logit = Xh_t @ self.OutputWeights + self.OutputBias # Y_hat
            outputs[t, :] = logit 
        
        # Return the hidden state for grad descent, output of the RNN 
        return hidden, outputs[-1] 

    def backward_pass():
        pass 

    def train(self, X: torch.tensor, Y: torch.tensor):
        for _ in range(self.epochs): # e = number of epoch we are at
            for index, sequence in enumerate(X[:self.epochs]):
                hidden, y_pred = self.forward_pass(sequence, Y[index])
                # print(f'sequence: {sequence}')
                # print(f'index: {index}, y value: {Y[index]}')
                # print(hidden)
                # print("-")
                # print(output) # Y_hat
                loss = 1/2 * ((Y[index] - y_pred) ** 2) # Squared error 
                print(f'loss: {loss}')
                

def prepare_input():
    df = pd.read_csv('data/day_data.csv')
    # Normalized temperature in Celsius
    temperatures = df['temp'].tolist()

    Xs, Ys = [], [] 

    for i in range(len(temperatures)-TIME_STEPS):
        Xs.append(temperatures[i:i+TIME_STEPS])
        Ys.append(temperatures[i+TIME_STEPS])

    Xs = torch.tensor(Xs)
    Ys = torch.tensor(Ys)

    # 80, 20 split 
    number_sequences = Xs.shape[0]
    train_size = int(number_sequences * 0.80)

    X = Xs[:train_size]
    Y = Ys[:train_size]

    testX = Xs[train_size:]
    testY = Ys[train_size:]

    return X, Y, testX, testY

def main():
    X, Y, testX, testY = prepare_input()
    network = RNN()
    network.train(X, Y)

if __name__ == "__main__":
    main()