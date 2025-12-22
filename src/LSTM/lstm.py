import torch
from cell_block import Cell
from typing import List

class LSTM: 
    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int, 
                 learning_rate: float, epochs: int, curr_version: int):
        self.cell_dims = hidden_dims
        self.output_dims = output_dims
        self.cell = Cell(input_dims, hidden_dims, output_dims)
        self.states_cache = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.version = curr_version

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
    
    def save_weights(self, path: str) -> None:
        torch.save(self.cell.get_state(), path)

    def sample(self, x_sequence: torch.tensor) -> torch.tensor:
        weights = torch.load(f"weights/lstm_v{self.version}.pt")
        # Update dimensions to match loaded weights
        self.cell_dims = weights["Wf"].shape[1]  # hidden_dims
        self.output_dims = weights["Wy"].shape[1]  # output_dims
        self.cell.input_dims = weights["Wf"].shape[0] - self.cell_dims  # input_dims
        self.cell.load_state(weights)
        prediction = self.forward(x_sequence)
        return prediction

    def train(self, X: torch.tensor, Y: torch.tensor, testX: torch.tensor, 
              testY: torch.tensor) -> List[float]:
        losses = [] 
        for e in range(self.epochs):
            epoch_loss = 0.0 
            for index, x_sequence in enumerate(X):
                y_pred = self.forward(x_sequence)
                y_true = Y[index].view(-1, 1)

                dZ = (2/y_pred.numel() * (y_pred - y_true))
                epoch_loss += self.calculate_loss(y_pred, y_true)

                # Zero out gradients before each backward pass
                self.cell.zero_grad()
                self.backward(dZ, x_sequence)
                self.sgd_step()

            valid_loss = 0
            for index, sequence in enumerate(testX):
                y_pred = self.forward(sequence)
                y_true = testY[index].view(-1, 1)
                valid_loss += self.calculate_loss(y_pred, y_true).item()
            valid_avg = valid_loss / len(testX)

            train_avg = epoch_loss / len(X)
            losses.append(train_avg.item())
            print(f'Epoch {e+1}, Train Loss: {train_avg.item()}, Test Loss: {valid_avg}')