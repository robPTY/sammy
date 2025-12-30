import torch
from typing import List, Tuple
from cell_block import Cell

class Decoder:
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_didm = embedding_dim
        self.embedding = torch.randn(vocab_size, embedding_dim) * 0.1
        self.dEmbeddings = torch.zeros_like(self.embedding)
        self.LSTM_cell = Cell(input_dims = embedding_dim, hidden_dims = hidden_dim,
                             output_dims = hidden_dim)
        # For backprop
        self.W_out = torch.randn(hidden_dim, vocab_size) * 0.1
        self.b_out = torch.zeros(1, vocab_size)
        self.dW_out = torch.zeros_like(self.W_out)
        self.db_out = torch.zeros_like(self.b_out)
        self.states_cache = [] 
    
    def softmax(self, x: torch.Tensor) -> torch.Tensor:
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
    
    def forward_step(self, token_id: int, cell_state: torch.tensor,
                     hidden_state: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        embedded = self.embedding[token_id].view(1, -1)  # (1, embedding_dim)
        Xt, ft, it, ct, ot, tanh_cs, new_cell, new_hidden, z = self.LSTM_cell.forward(
            cell_state, hidden_state, embedded
        )
        
        # Project to vocabulary (predict next token)
        logits = new_hidden @ self.W_out + self.b_out
        
        return logits, new_hidden, new_cell
    
    def forward(self, encoder_hidden: torch.tensor, target_tokens: List[int],
                encoder_cell: torch.tensor):
        self.states_cache = []
        T = len(target_tokens)

        hidden_state = encoder_hidden
        cell_state = encoder_cell
        logits = [] 

        for t in range(T-1):
            curr_token = target_tokens[t]
            logit, hidden_state, cell_state = self.forward_step(curr_token, cell_state, hidden_state)
            logits.append(logit)

            self.states_cache.append({
                "token_id": curr_token, "hidden": hidden_state, "cell_state": cell_state,
                "logit": logit
            })
            
        return torch.cat(logits, dim=0)

    def generate(self, encoder_hidden: torch.tensor, encoder_cell: torch.tensor,
                 sos_token: int, eos_token: int, max_length: int = 50) -> List[int]:
        hidden_state = encoder_hidden
        cell_state = encoder_cell 

        output = [sos_token]
        curr_token = sos_token

        for _ in range(max_length):
            logits, hidden_state, cell_state = self.forward_step(curr_token, cell_state, hidden_state)
            probabilities = self.softmax(logits)
            next_token = torch.argmax(probabilities, dim=-1).item()
            
            output.append(next_token)
            if next_token == eos_token:
                break
            curr_token = next_token  # Fixed: was "current_token"
        
        return output