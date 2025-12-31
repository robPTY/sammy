import torch
import pandas as pd
from typing import Tuple, List
from tokenizer import Tokenizer
from encoder import Encoder
from decoder import Decoder

Sets = Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]

def load_dataset(path: str) -> Sets:
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["english", "spanish", "meta"])
    df = df[["english", "spanish"]]

    Xs = df["english"]
    Ys = df["spanish"]

    # 80, 20 split
    vocab_size = Xs.shape[0]
    train_size = int(vocab_size * 0.80)

    X_train = Xs[:train_size]
    Y_train = Ys[:train_size]
    X_test = Xs[train_size:]
    Y_test = Ys[train_size:]

    return X_train, Y_train, X_test, Y_test

def cross_entropy_loss(logits: torch.tensor, target_tokens: List[int]) -> Tuple[float, torch.tensor]:
    T = len(target_tokens) - 1
    d_logits = torch.zeros_like(logits)
    total_loss = 0.0

    for t in range(T):
        logit = logits[t]
        exp_logit = torch.exp(logit - torch.max(logit))
        probs = exp_logit / torch.sum(exp_logit)

        target_token = target_tokens[t+1]
        total_loss += -torch.log(probs[target_token] + 1e-9).item()

        d_logits[t] = probs
        d_logits[t][target_token] -= 1
      
    avg_loss = total_loss / T
    d_logits = d_logits / T 
    
    return avg_loss, d_logits

def main():
    VOCAB_SIZE = 300
    file_path = "data/eng_to_spa.txt"
    X_train, Y_train, X_test, Y_test = load_dataset(file_path)
    
    print('Training tokenizer on corpora...')
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenized_tokens = tokenizer.tokenize(X_train[:1000])
    print(f'Learned {len(tokenizer.merges)} BPE merges')

    encoder = Encoder(vocab_size = VOCAB_SIZE + 2, embedding_dim = 64, hidden_dim = 128)
    decoder = Decoder(vocab_size = VOCAB_SIZE + 2, embedding_dim = 64, hidden_dim = 128)

    num_examples = 10
    
    for i in range(50, 50 + num_examples):
        english = X_train.iloc[i]
        spanish = Y_train.iloc[i]

        source_tokens = tokenizer.encode(english)
        target_tokens = tokenizer.encode(spanish)

        # Encode
        hidden, cell = encoder.encode(source_tokens)
        # Decode
        logits = decoder.forward(hidden, target_tokens, cell)
        
        # Compute loss and gradient
        loss, d_logits = cross_entropy_loss(decoder, logits, target_tokens)
        
        print(f'Example {i}: loss = {loss:.4f}')

    return 1

if __name__ == "__main__":
    main()