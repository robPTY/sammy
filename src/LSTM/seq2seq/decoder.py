import torch
from typing import List, Tuple
from LSTM.seq2seq.seq2seqcell import S2SCell

class Decoder:
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding = torch.randn(vocab_size, embedding_dim) * 0.1
        self.dEmbeddings = torch.zeros_like(self.embedding)
        self.LSTM_cell = S2SCell(input_dims = embedding_dim, hidden_dims = hidden_dim,
                             output_dims = hidden_dim)
        # For backprop
        self.W_out = torch.randn(hidden_dim, vocab_size) * 0.1
        self.b_out = torch.zeros(1, vocab_size)
        self.dW_out = torch.zeros_like(self.W_out)
        self.db_out = torch.zeros_like(self.b_out)
        self.states_cache = [] 
    
    def softmax(self, x: torch.Tensor) -> torch.tensor:
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
    
    def forward_step(self, token_id: int, cell_state: torch.tensor, hidden_state: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        embedded = self.embedding[token_id].view(1, -1)  # (1, embedding_dim)
        Xt, ft, it, ct, ot, tanh_cs, new_cell, new_hidden = self.LSTM_cell.forward(cell_state, hidden_state, embedded)

        cache_entry = {
            "Xt": Xt, "ft": ft, "it": it, "ot": ot, "candidate": ct,
            "tanh_cell_state": tanh_cs, "cell_state": new_cell,
            "prev_cell_state": cell_state, "prev_hidden": hidden_state,
            "hidden_state": new_hidden
        }
        
        # Project to vocabulary (predict next token)
        logits = new_hidden @ self.W_out + self.b_out
        
        return logits, new_hidden, new_cell, cache_entry
    
    def forward(self, encoder_hidden: torch.tensor, target_tokens: List[int],
                encoder_cell: torch.tensor) -> torch.tensor:
        self.states_cache = []
        T = len(target_tokens)

        hidden_state = encoder_hidden
        cell_state = encoder_cell
        logits = [] 

        for t in range(T-1):
            curr_token = target_tokens[t]
            logit, hidden_state, cell_state, cache_entry = self.forward_step(curr_token, cell_state, hidden_state)
            logits.append(logit)

            cache_entry["token"] = curr_token
            cache_entry["logit"] = logit
            self.states_cache.append(cache_entry)
            
        return torch.cat(logits, dim=0)
    
    def backward(self, d_logits: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        T = len(self.states_cache)
        d_hidden_next = torch.zeros(1, self.hidden_dim)
        d_cell_next = torch.zeros(1, self.hidden_dim)

        self.zero_grad()

        for t in range(T-1, -1, -1):
            # print(f'time {t} with grads of {d_hidden_next} and {d_cell_next}')
            d_logit_t = d_logits[t].view(1, -1) # logit at time t
            state_at_t = self.states_cache[t]
            hidden_t = state_at_t['hidden_state']
            token_id = state_at_t['token']

            self.dW_out += hidden_t.T @ d_logit_t # dL/dWout 
            self.db_out += d_logit_t # dL/db_out

            dhidden_dout = d_logit_t @ self.W_out.T # dL/dHt (from output)
            dhidden = dhidden_dout + d_hidden_next # dL/dHt + Ht+1

            dx, dprev_hidden, dprev_cell = self.LSTM_cell.backward(state_at_t, dhidden, d_cell_next)

            d_hidden_next = dprev_hidden
            d_cell_next = dprev_cell

            self.dEmbeddings[token_id] += dx.squeeze()

        return d_cell_next, d_hidden_next


    def generate(self, encoder_hidden: torch.tensor, encoder_cell: torch.tensor,
                 sos_token: int, eos_token: int, max_length: int = 50) -> List[int]:
        hidden_state = encoder_hidden
        cell_state = encoder_cell 

        output = [sos_token]
        curr_token = sos_token

        for _ in range(max_length):
            logits, hidden_state, cell_state, _ = self.forward_step(curr_token, cell_state, hidden_state)
            probabilities = self.softmax(logits)
            next_token = torch.argmax(probabilities, dim=-1).item()
            
            output.append(next_token)
            if next_token == eos_token:
                break
            curr_token = next_token
        
        return output
    
    def zero_grad(self) -> None:
        self.dW_out.zero_()
        self.db_out.zero_()
        self.dEmbeddings.zero_()
        self.LSTM_cell.zero_grad()

    def sgd_step(self, learning_rate: float) -> None:
        self.W_out -= learning_rate * self.dW_out
        self.b_out -= learning_rate * self.db_out
        self.embedding -= learning_rate * self.dEmbeddings
        self.LSTM_cell.sgd_step(learning_rate)

    def save_weights(self, path: str) -> None:
        torch.save({
            'embedding': self.embedding,
            'lstm': self.LSTM_cell.get_state(),
            'W_out': self.W_out,
            'b_out': self.b_out,
        }, path)

    def load_weights(self, path: str) -> None:
        state = torch.load(path)
        self.embedding = state['embedding']
        self.LSTM_cell.load_state(state['lstm'])
        self.W_out = state['W_out']
        self.b_out = state['b_out']