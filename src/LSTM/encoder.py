import torch
from typing import List, Tuple
from lstm import LSTM

class Encoder:
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = torch.randn(vocab_size, embedding_dim) * 0.1
        self.LSTM = LSTM(input_dims=embedding_dim, hidden_dims=hidden_dim,
        output_dims=hidden_dim, learning_rate=0.01, epochs=50, curr_version=1)
    
    def encode(self, tokens: List[int]) -> Tuple[torch.tensor, torch.tensor]:
        tokens_tensor = torch.tensor(tokens)
        embedded = self.embedding[tokens_tensor]

        z = self.LSTM.forward(embedded)
        final_state = self.LSTM.states_cache[-1]
        final_hidden = final_state["ht"]
        final_cell_state = final_state["cell_state"]

        return final_hidden, final_cell_state
        
    def backward(self, dhidden: torch.tensor, dcell: torch.tensor) -> None:
        T = 11 
        next_dhidden, next_dcell = dhidden, dcell
        for t in range(T-1, -1, -1):
            pass