import torch
from rnn import RNN
from .AutoRNN import AutoRNN

def check_close(name, manual, auto):
    ok = torch.allclose(manual, auto, atol=1e-4, rtol=1e-4)
    print(f"{name}: {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("manual:", manual)
        print("auto:", auto)

# Test manual gradiants against PyTorch autograd
def test_manual_grad():
    torch.manual_seed(0)
    
    # Build model
    manual = RNN()
    auto = AutoRNN(manual)

    sequence = torch.tensor([0.220833, 0.134783, 0.144348])
    # Run forward pass 
    hidden, out = manual.forward_pass(sequence)
    target = torch.tensor([0.189091])
    dy = (out[-1] - target).view(1,1)
    dWy, dB, dWh, dWx, dG = manual.backward_pass(sequence, hidden, dy)

    auto.zero_grad()
    y_pred = auto(sequence)
    loss = 0.5 * (y_pred - target).pow(2).sum()
    loss.backward()
    
    check_close("InputWeights", dWx, auto.Wx.grad)
    check_close("HiddenWeights", dWh, auto.Wh.grad)
    check_close("OutputWeights", dWy, auto.Wy.grad)
    check_close("HiddenBias", dG, auto.bh.grad)
    check_close("OutputBias", dB, auto.by.grad)
    return 

# Test the networks train loss vs networks test loss

def main():
    test_manual_grad()
    return 1

if __name__ == "__main__":
    main() 