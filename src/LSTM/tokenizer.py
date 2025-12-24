import pandas as pd
from typing import Dict, List, Tuple

class Tokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def get_pairs(self, tokens: List[int]) -> Dict:
        pairs = {} 
        for i in range(1, len(tokens)):
            token_pair = (tokens[i-1], tokens[i])
            pairs[token_pair] = 1 + pairs.get(token_pair, 0)
        return pairs 
    
    def merge(self, tokens: List[int], target: Tuple[int, int], new_index: int) -> List[int]:
        new_tokens = []
        index = 0
        while index < len(tokens):
            a, b = tokens[index], tokens[index+1]
            if index < len(tokens) - 1 and a == target[0] and b == target[1]:
                new_tokens.append(new_index)
                index += 2
            else:
                new_tokens.append(tokens[index])
                index += 1
        return new_tokens

    def byte_pair_enc(self, X: pd.DataFrame) -> List[int]:
        tokens = [] 
        for x_seq in X:
            tokens.extend(list(map(int, x_seq.encode('utf-8'))))
        return tokens

    def tokenize(self, X: pd.DataFrame) -> List[int]:
        byte_length = 256
        number_of_merges = self.vocab_size - byte_length
        tokens = self.byte_pair_enc(X)
        merges = {} # (int, int) -> int 
        for i in range(number_of_merges):
            pairs = self.get_pairs(tokens)
            top_pair = max(pairs, key=pairs.get)
            tokens = self.merge(tokens, top_pair, i + byte_length)
            merges[top_pair] = i + byte_length
        return tokens