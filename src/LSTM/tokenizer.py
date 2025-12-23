import torch 
from typing import Dict

class Tokenizer:
    def __init__(self):
        pass 

    def tokenize(self, X: torch.tensor) -> Dict:
        for x_sequence in X[:2]:
            print(x_sequence)