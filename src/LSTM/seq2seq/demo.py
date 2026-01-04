import torch
import pandas as pd
import sys
sys.path.append('..')

from tokenizer import Tokenizer
from encoder import Encoder
from decoder import Decoder

VOCAB_SIZE = 300
EMBEDDING_DIMS = 64
HIDDEN_DIMS = 128

def load_model():
    print("Loading model...")
    
    # Train tokenizer on corpus
    df = pd.read_csv("data/eng_to_spa.txt", sep="\t", header=None, names=["english", "spanish", "meta"])
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.tokenize(df["english"][:5000])
    
    # Load encoder & decoder
    encoder = Encoder(vocab_size=VOCAB_SIZE + 2, embedding_dim=EMBEDDING_DIMS, hidden_dim=HIDDEN_DIMS)
    decoder = Decoder(vocab_size=VOCAB_SIZE + 2, embedding_dim=EMBEDDING_DIMS, hidden_dim=HIDDEN_DIMS)
    encoder.load_weights('weights/encoder_final.pt')
    decoder.load_weights('weights/decoder_final.pt')
    
    print("Ready!\n")
    return tokenizer, encoder, decoder

def translate(text, tokenizer, encoder, decoder):
    tokens = tokenizer.encode(text)
    hidden, cell = encoder.encode(tokens)
    output = decoder.generate(hidden, cell, tokenizer.SOS, tokenizer.EOS, max_length=50)
    return tokenizer.decode(output)

def main():
    print("Seq2Seq Translator (English → Spanish)")
    tokenizer, encoder, decoder = load_model()
    
    print("Type English text to translate. Type 'quit' or 'q' to exit.\n")
    
    while True:
        text = input("English: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\n¡Adiós!\n")
            break
        
        if not text:
            continue
        
        spanish = translate(text, tokenizer, encoder, decoder)
        print(f"Spanish: {spanish}\n")

if __name__ == "__main__":
    main()
