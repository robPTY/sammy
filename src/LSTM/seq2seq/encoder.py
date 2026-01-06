import torch
from typing import List, Tuple
from seq2seqcell import S2SCell

class Encoder:
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = torch.randn(vocab_size, embedding_dim) * 0.1
        self.dEmbeddings = torch.zeros_like(self.embedding)
        self.LSTM_cell = S2SCell(input_dims=embedding_dim, hidden_dims=hidden_dim,
                                  output_dims=hidden_dim)
        self.states_cache = []
        self.tokens = []
    
    def encode(self, tokens: List[int]) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        all_hidden_states = []
        self.states_cache = []
        self.tokens = tokens
        T = len(tokens)
        
        hidden_state = torch.zeros(1, self.hidden_dim)
        cell_state = torch.zeros(1, self.hidden_dim)

        for t in range(T):
            token_id = tokens[t]
            embedded_tokens = self.embedding[token_id].view(1, -1)
            
            Xt, ft, it, ct, ot, tanh_cs, new_cell, new_hidden = self.LSTM_cell.forward(cell_state, hidden_state, embedded_tokens)
            all_hidden_states.append(new_hidden)
            
            self.states_cache.append({
                "token_id": token_id,
                "Xt": Xt, "ft": ft, "it": it, "candidate": ct, "ot": ot,
                "tanh_cell_state": tanh_cs, "cell_state": new_cell,
                "prev_cell_state": cell_state, "prev_hidden": hidden_state,
                "hidden_state": new_hidden
            })
            
            hidden_state = new_hidden
            cell_state = new_cell
        
        encoder_outputs = torch.stack(all_hidden_states)

        return encoder_outputs, hidden_state, cell_state
        
    def backward(self, dhidden: torch.tensor, dcell: torch.tensor) -> None:
        T = len(self.states_cache)
        next_dhidden, next_dcell = dhidden, dcell
        
        for t in range(T-1, -1, -1):
            cache = self.states_cache[t]
            token_id = cache["token_id"]
            
            dx, dprev_hidden, dprev_cell = self.LSTM_cell.backward(cache, next_dhidden, next_dcell)
            
            self.dEmbeddings[token_id] += dx.squeeze()
            
            next_dhidden = dprev_hidden
            next_dcell = dprev_cell
    
    def zero_grad(self) -> None:
        self.dEmbeddings.zero_()
        self.LSTM_cell.zero_grad()
    
    def sgd_step(self, learning_rate: float) -> None:
        self.embedding -= learning_rate * self.dEmbeddings
        self.LSTM_cell.sgd_step(learning_rate)

    def save_weights(self, path: str) -> None:
        torch.save({
            'embedding': self.embedding,
            'lstm': self.LSTM_cell.get_state(),
        }, path)

    def load_weights(self, path: str) -> None:
        state = torch.load(path)
        self.embedding = state['embedding']
        self.LSTM_cell.load_state(state['lstm'])