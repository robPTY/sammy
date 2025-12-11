import torch
from cell_block import Cell

class LSTM: 
    def __init__(self, input_dims, hidden_dims):
        self.cell_dims = hidden_dims
        self.cell = Cell(input_dims, hidden_dims)

    def forward(self, x_sequence: torch.tensor):
        T = len(x_sequence)

        hiddens = torch.zeros(1, self.cell_dims)
        cell_state = torch.zeros(1, self.cell_dims)

        prev_hidden = None
        for t in range(T):
            xt = x_sequence[t].view((1, -1))
            hidden, cell_state = self.cell.forward(prev_hidden, xt)

    def train(self, X: torch.tensor):
        for index, x_sequence in enumerate(X):
            self.forward(x_sequence)