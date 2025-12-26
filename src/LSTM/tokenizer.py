import pandas as pd
from typing import Dict, List, Tuple

class Tokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.SOS, self.EOS = vocab_size, vocab_size + 1
        self.tokens = []
        self.merges = {}
    
    def get_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)} 
        if len(self.tokens) > 0:
            for (p0, p1), index in self.merges.items():
                vocab[index] = vocab[p0] + vocab[p1]
        return vocab 

    def get_pairs(self, tokens: List[int]) -> Dict:
        pairs = {} 
        for i in range(1, len(tokens)):
            if tokens[i-1] >= 256 or tokens[i] >= 256:
                continue    
            token_pair = (tokens[i-1], tokens[i])
            pairs[token_pair] = 1 + pairs.get(token_pair, 0)
        return pairs 
    
    def merge(self, tokens: List[int], target: Tuple[int, int], new_index: int) -> List[int]:
        new_tokens = []
        index = 0
        while index < len(tokens):
            if index < len(tokens) - 1 and tokens[index] == target[0] and tokens[index+1] == target[1]:
                new_tokens.append(new_index)
                index += 2
            else:
                new_tokens.append(tokens[index])
                index += 1
        return new_tokens

    def byte_pair_enc(self, X: pd.DataFrame) -> List[int]:
        tokens = [] 
        for x_seq in X:
            tokens.append(self.SOS)
            tokens.extend(list(map(int, x_seq.encode('utf-8'))))
            tokens.append(self.EOS)
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

        self.tokens = tokens
        self.merges = merges
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        vocab = self.get_vocab()
        to_decode = b"".join(vocab[token] for token in tokens if token < 256)
        text = to_decode.decode('utf-8', errors="replace")
        return text
    
    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode('utf-8'))