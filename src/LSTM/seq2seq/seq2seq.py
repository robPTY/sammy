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
    VOCAB_SIZE = 1000
    file_path = "data/eng_to_spa.txt"
    X_train, Y_train, X_test, Y_test = load_dataset(file_path)
    
    shuffle_idx = torch.randperm(len(X_train)).tolist()
    X_train = X_train.iloc[shuffle_idx].reset_index(drop=True)
    Y_train = Y_train.iloc[shuffle_idx].reset_index(drop=True)
    
    print('Training tokenizer on corpora...')
    combined_corpus = pd.concat([X_train, Y_train])
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenized_tokens = tokenizer.tokenize(combined_corpus)
    print(f'Learned {len(tokenizer.merges)} BPE merges')

    print('Training encoder and decoder...')    

    encoder = Encoder(vocab_size = VOCAB_SIZE + 2, embedding_dim = 256, hidden_dim = 512)
    decoder = Decoder(vocab_size = VOCAB_SIZE + 2, embedding_dim = 256, hidden_dim = 512)

    NUM_EPOCHS = 30
    LEARNING_RATE = 0.005
    NUM_TRAIN = min(50000, len(X_train))
    NUM_VAL = min(500, len(X_test))
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        
        for i in range(NUM_TRAIN):
            english = X_train.iloc[i]
            spanish = Y_train.iloc[i]

            # Tokenize input
            source_tokens = tokenizer.encode(english)
            target_tokens = tokenizer.encode(spanish)

            # Forward pass
            encoder_outputs, hidden, cell = encoder.encode(source_tokens)
            logits = decoder.forward(encoder_outputs, hidden, target_tokens, cell)
            
            # Compute loss and gradient
            loss, d_logits = cross_entropy_loss(logits, target_tokens)
            epoch_loss += loss
            
            # Backward passes
            encoder.zero_grad()
            dcell_state, dhidden_state = decoder.backward(d_logits, encoder_outputs)
            encoder.backward(dhidden_state, dcell_state)

            # SGD step
            encoder.sgd_step(LEARNING_RATE)
            decoder.sgd_step(LEARNING_RATE)
        
        train_loss = epoch_loss / NUM_TRAIN
        val_loss = 0.0
        for i in range(NUM_VAL):
            english = X_test.iloc[i]
            spanish = Y_test.iloc[i]
            
            source_tokens = tokenizer.encode(english)
            target_tokens = tokenizer.encode(spanish)
            
            hidden, cell = encoder.encode(source_tokens)
            logits = decoder.forward(hidden, target_tokens, cell)
            
            loss, _ = cross_entropy_loss(logits, target_tokens)
            val_loss += loss
        
        val_loss = val_loss / NUM_VAL
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}')

    encoder.save_weights('weights/encoder_v2.pt')
    decoder.save_weights('weights/decoder_v2.pt')
    print("Weights saved!")

    print("\n--- Generation Test ---")
    for i in range(500, 505):
        english = X_train.iloc[i]
        spanish_actual = Y_train.iloc[i]
        source_tokens = tokenizer.encode(english)
        hidden, cell = encoder.encode(source_tokens)
        
        # Generate
        generated_tokens = decoder.generate(
            hidden, cell,
            sos_token=tokenizer.SOS,
            eos_token=tokenizer.EOS,
            max_length=30
        )
        generated_text = tokenizer.decode(generated_tokens)
        
        print(f'\nEnglish: {english}')
        print(f'Actual:  {spanish_actual}')
        print(f'Generated: {generated_text}')

    return 1

if __name__ == "__main__":
    main()