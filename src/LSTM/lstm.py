import torch
from cell_block import Cell

class LSTM: 
    def __init__(self, input_dims, hidden_dims):
        self.cell_dims = hidden_dims
        self.cell = Cell(input_dims, hidden_dims)

    def forward(self, x_sequence: torch.tensor):
        T = len(x_sequence)

        prev_hidden = torch.zeros(1, self.cell_dims)
        cell_state = torch.zeros(1, self.cell_dims)

        for t in range(T):
            xt = x_sequence[t].view((1, -1))
            self.cell.forward(cell_state, prev_hidden, xt)
        
        return prev_hidden, cell_state

    def train(self, X: torch.tensor):
        for index, x_sequence in enumerate(X[:1]):
            self.forward(x_sequence)