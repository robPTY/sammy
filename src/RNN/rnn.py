import torch
import pandas as pd
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

    def forward_pass(network, temperatures):
        previous_step = None
        logits = torch.zeros((100, 1))
        for t in range(100):
            # Input layer
            X = torch.tensor(temperatures[t]).unsqueeze((0))
            X_i = X @ network.InputWeights

            # Hidden layer
            if previous_step is None: 
                Xh_t = F.relu(X_i)
            else:
                X_h = previous_step @ network.HiddenWeights
                Xh_t = F.tanh(X_i + X_h + network.HiddenBias)
            previous_step = Xh_t

            print(f'hidden state: {Xh_t}')
            
            # Output layer
            logit = Xh_t @ network.OutputWeights + network.OutputBias
            logits[t] = logit
            print(f'at time t={t}, temperature is {logit}')
        
        counts = logits.exp() 
        dimension = 1
        probabilities = counts / counts.sum(dimension, keepdim=True)
        return 

    def backward_pass():
        pass

def prepare_input():
    df = pd.read_csv('data/day_data.csv')
    # Normalized temperature in Celsius
    temperatures = df['temp'].tolist()
    return temperatures[:100]

def main():
    temperatures = prepare_input()
    network = RNN()
    RNN.forward_pass(network, temperatures)

if __name__ == "__main__":
    main()