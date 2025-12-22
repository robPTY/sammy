import torch
import pandas as pd
from lstm import LSTM

def load_data(seq_length: int, pred_length: int):
    file_name = "data/Sunspots.csv"
    df = pd.read_csv(file_name)

    sunspots = df['Monthly Mean Total Sunspot Number'].tolist()

    Xs, Ys = [], [] 
    for index in range(len(sunspots) - seq_length - pred_length + 1):
        Xs.append(sunspots[index:index+seq_length])
        Ys.append(sunspots[index+seq_length:index+seq_length+pred_length])

    Xs = torch.tensor(Xs)
    Ys = torch.tensor(Ys)

    # 80, 20 split 
    number_sequences = Xs.shape[0]
    train_size = int(number_sequences * 0.80)

    X = Xs[:train_size]
    Y = Ys[:train_size]

    testX = Xs[train_size:]
    testY = Ys[train_size:]
    
    # Normalize inputs
    mean = X.mean()
    standard_deviation = X.std()

    X_train = (X - mean) / standard_deviation
    Y_train = (Y - mean) / standard_deviation

    X_test = (testX - mean) / standard_deviation
    Y_test = (testY - mean) / standard_deviation   
    
    return X_train, Y_train, X_test, Y_test, mean, standard_deviation

def main():
    SEQ_LENGTH, PRED_LENGTH = 5, 5
    INPUT_SIZE, HIDDEN_SIZE = 1, 5
    OUTPUT_SIZE = 1
    LEARNING_RATE, EPOCHS = 0.01, 75
    CURR_VERSION = 3

    X, Y, testX, testY, mean, std = load_data(SEQ_LENGTH, PRED_LENGTH)
    network = LSTM(input_dims=INPUT_SIZE, hidden_dims=HIDDEN_SIZE, 
                   output_dims=OUTPUT_SIZE, learning_rate=LEARNING_RATE,
                   epochs=EPOCHS, curr_version=CURR_VERSION)
    
    # Train model
    network.train(X, Y, testX, testY)

    # Save weights of most recently trained
    network.save_weights(f'weights/lstm_v{CURR_VERSION}.pt')

    print("\n" + "=" * 50)
    print("SUNSPOT PREDICTIONS")
    print("=" * 50)

    # Show 3 example sequences
    for i in [0, 50, 100]:
        x_sample = X[i]
        y_actual = Y[i]
        y_pred = network.sample(x_sample)

        # Un-normalize vals to get real temps
        x_real = (x_sample * std + mean).tolist()
        y_actual_real = (y_actual * std + mean).tolist()
        y_pred_real = (y_pred.squeeze() * std + mean).tolist()

        print(f"\nExample {i + 1}:")
        print(f"  Input:      {[f'{v:.1f}' for v in x_real]}")
        print(f"  Actual:     {[f'{v:.1f}' for v in y_actual_real]}")
        print(f"  Predicted:  {[f'{v:.1f}' for v in y_pred_real]}")

    print("\n" + "=" * 50)
    
    return 1

if __name__ == "__main__":
    main()