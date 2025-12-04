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
    print(f'prediction before training: {network.predict([0.220833, 0.134783, 0.144348])}')
    network.train(X, Y, testX, testY)
    print(f'prediction after training: {network.predict([0.220833, 0.134783, 0.144348])}')
    print(f'correct value: 0.189091')
    print(f'prediction after training: {network.predict([0.266087, 0.318261, 0.435833])}')
    print(f'correct value: 0.521667')
    print(f'prediction after training: {network.predict([0.399167, 0.285217, 0.303333])}')
    print(f'correct value: 0.182222')

if __name__ == "__main__":
    main()