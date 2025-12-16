import torch
from cell_block import Cell

class LSTM: 
    def __init__(self, input_dims, hidden_dims, output_dims):
        self.cell_dims = hidden_dims
        self.output_dims = output_dims
        self.cell = Cell(input_dims, hidden_dims, output_dims)
        # self.Xt_hist = torch.zeros(T, input_dims + hidden_dims)
        # self.ft_hist = torch.zeros(T, hidden_dims) 
        # self.it_hist = torch.zeros(T, hidden_dims) 
        # self.ct_hist = torch.zeros(T, hidden_dims)  
        # self.ot_hist = torch.zeros(T, hidden_dims)  
        # self.cell_hist = torch.zeros(T, hidden_dims) 
        # self.h_hist = torch.zeros(T, hidden_dims)

    def forward(self, x_sequence: torch.tensor):
        T = x_sequence.shape[0]

        prev_hidden = torch.zeros(1, self.cell_dims)
        cell_state = torch.zeros(1, self.cell_dims)
        z = torch.zeros(T, self.output_dims)

        for t in range(T):
            xt = x_sequence[t].view((1, -1))
            Xt, ft, it, ct, ot, new_cell_state, hidden, zt = self.cell.forward(cell_state, prev_hidden, xt)
            z[t] = zt.squeeze(0)
            
            # print(f'cell state at time {t}: {cell_state}')
            # print(f'hidden_state at time {t}: {hidden}')
            prev_hidden = hidden
            cell_state = new_cell_state
        
        return Xt, ft, it, ct, ot, cell_state, prev_hidden, z
    
    def backward(self, Xt: torch.tensor,ot: torch.tensor, cell_state: torch.tensor, 
                 hidden_state: torch.tensor, dZ: torch.tensor, x_sequence: torch.tensor):
        self.cell.backward(Xt, ot, cell_state, hidden_state, dZ, x_sequence)
    
    def calculate_loss(self, Y_pred: torch.tensor, Y_true: torch.tensor) -> torch.tensor:
        N = Y_pred.numel()
        return torch.sum((Y_true - Y_pred)**2) / N

    def train(self, X: torch.tensor, Y: torch.tensor):
        for index, x_sequence in enumerate(X[:1]):
            Xt, ft, it, ct, ot, cell_state, prev_hidden, y_pred = self.forward(x_sequence)
            y_true = Y[index].view(-1, 1)

            # print(x_sequence)
            # print(y_true)
            # print(y_pred)

            dZ = (2/y_pred.numel() * (y_pred - y_true))
            loss = self.calculate_loss(y_pred, y_true)
            print(f'loss: {loss.item()}')
            self.backward(Xt, ot, cell_state, prev_hidden, dZ, x_sequence)