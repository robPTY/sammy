import torch
import pandas as pd
from typing import Tuple
from tokenizer import Tokenizer

Sets = Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]

def load_dataset(path: str) -> Sets:
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["english", "spanish", "meta"])
    df = df[["english", "spanish"]]

    Xs = df["english"]
    Ys = df["spanish"]

    Xs = torch.tensor(Xs)
    Ys = torch.tensor(Ys)

    # 80, 20 split
    vocab_size = Xs.shape[0]
    train_size = vocab_size * 0.80

    X_train = Xs[:train_size]
    Y_train = Ys[:train_size]
    X_test = Xs[train_size:]
    Y_test = Ys[train_size:]

    return X_train, Y_train, X_test, Y_test

def main():
    file_path = "data/eng_to_spa.txt"
    X_train, Y_train, X_test, Y_test = load_dataset(file_path)
    
    # Tokenize the inputs
    tokenizer = Tokenizer()
    tokenizer.tokenize(X_train)

    return 1

if __name__ == "__main__":
    main()