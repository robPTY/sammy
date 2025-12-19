import torch
from cell_block import Cell
from typing import List, Dict

class LSTM: 
    def __init__(self, input_dims, hidden_dims, output_dims, learning_rate):
        self.cell_dims = hidden_dims
        self.output_dims = output_dims
        self.cell = Cell(input_dims, hidden_dims, output_dims)
        self.states_cache = []
        self.learning_rate = learning_rate

    def forward(self, x_sequence: torch.tensor):
        self.states_cache = [] 
        T = x_sequence.shape[0]

        prev_hidden = torch.zeros(1, self.cell_dims)
        cell_state = torch.zeros(1, self.cell_dims)
        z = torch.zeros(T, self.output_dims)

        for t in range(T):
            xt = x_sequence[t].view((1, -1))
            Xt, ft, it, ct, ot, tanh_cell_state, new_cell_state, hidden, zt = self.cell.forward(cell_state, prev_hidden, xt)
            z[t] = zt.squeeze(0)
            
            self.states_cache.append({
                "Xt": Xt, "ft": ft, "it": it, "candidate": ct, "ot": ot,
                "prev_cst": cell_state, "cell_state": new_cell_state,
                "tanh_cell_state": tanh_cell_state,
                "prev_hidden": prev_hidden, "ht": hidden
            })

            prev_hidden = hidden
            cell_state = new_cell_state

        return z
    
    def backward(self, dZ: torch.tensor, x_sequence: torch.tensor) -> None:
        T = x_sequence.shape[0]
        next_dht = torch.zeros(1, self.cell_dims) # dL/dHt+1
        next_dcst = torch.zeros(1, self.cell_dims) # dL/dCst+1

        for t in range(T-1, -1, -1):
            dXt, dprev_hidden, dprev_cell_state = self.cell.backward(self.states_cache[t], dZ[t], next_dht, next_dcst)

            next_dht = dprev_hidden
            next_dcst = dprev_cell_state

    def sgd_step(self) -> None:
        self.cell.sgd_step(self.learning_rate)
    
    def calculate_loss(self, Y_pred: torch.tensor, Y_true: torch.tensor) -> torch.tensor:
        N = Y_pred.numel()
        return torch.sum((Y_true - Y_pred)**2) / N

    def train(self, X: torch.tensor, Y: torch.tensor):
        for index, x_sequence in enumerate(X[:50]):
            y_pred = self.forward(x_sequence)
            y_true = Y[index].view(-1, 1)

            # print(x_sequence)
            # print(y_true)
            # print(y_pred)

            dZ = (2/y_pred.numel() * (y_pred - y_true))
            loss = self.calculate_loss(y_pred, y_true)
            print(f'loss: {loss.item()}')

            # Zero out gradients before each backward pass
            self.cell.zero_grad()
            self.backward(dZ, x_sequence)
            self.sgd_step()