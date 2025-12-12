import torch
from gates import ForgetGate, InputGate

class Cell:
    def __init__(self, input_dims, hidden_dims):
        # Forget gate
        self.Wf = torch.randn((input_dims + hidden_dims, hidden_dims)) * 0.01
        self.bf = torch.randn((1, hidden_dims))
        # Input gate
        self.Wi = torch.randn((input_dims + hidden_dims, hidden_dims)) * 0.01
        self.bi = torch.randn((1, hidden_dims))

    def forward(self, prev_ct: torch.tensor, prev_h: torch.tensor, x: torch.tensor):
        Xt = torch.cat((x, prev_h), dim=1)
        Ft = ForgetGate() 
        ft = Ft.forward(self.Wf, Xt, self.bf)

        cell_state = prev_ct * ft

        print(f'ft: {ft}')
        print(f'shape of ft :{ft.shape}')
        print(f'cell state: {cell_state}')

        It = InputGate()
        it = It.forward(self.Wi, Xt, self.bi)

