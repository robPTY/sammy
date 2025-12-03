import pandas as pd 
import torch
from rnn import RNN

TIME_STEPS = 3

def prepare_input():
    df = pd.read_csv('data/day_data.csv')
    # Normalized temperature in Celsius
    temperatures = df['temp'].tolist()

    Xs, Ys = [], [] 

    for i in range(len(temperatures)-TIME_STEPS):
        Xs.append(temperatures[i:i+TIME_STEPS])
        Ys.append(temperatures[i+TIME_STEPS])

    Xs = torch.tensor(Xs)
    Ys = torch.tensor(Ys)

    # 80, 20 split 
    number_sequences = Xs.shape[0]
    train_size = int(number_sequences * 0.80)

    X = Xs[:train_size]
    Y = Ys[:train_size]

    testX = Xs[train_size:]
    testY = Ys[train_size:]

    return X, Y, testX, testY

def main():
    X, Y, testX, testY = prepare_input()
    network = RNN() 
    network.train(X, Y)

if __name__ == "__main__":
    main()