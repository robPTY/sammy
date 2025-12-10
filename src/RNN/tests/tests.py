import torch
from RNN.rnn import RNN
from RNN.tests.AutoRNN import AutoRNN

def check_close(name, manual, auto):
    ok = torch.allclose(manual, auto, atol=1e-4, rtol=1e-4)
    print(f"{name}: {'OK' if ok else 'MISMATCH'}")
    print("manual:", manual)
    print("auto:", auto)

# Test manual gradiants against PyTorch autograd
def test_manual_grad():
    torch.manual_seed(0)
    
    # Build model
    manual = RNN()
    auto = AutoRNN(manual)

    with torch.no_grad():
        auto.Wx.data = manual.InputWeights.data.clone()
        auto.Wh.data = manual.HiddenWeights.data.clone()
        auto.Wy.data = manual.OutputWeights.data.clone()
        auto.bh.data = manual.HiddenBias.data.clone() 
        auto.by.data = manual.OutputBias.data.clone()


    sequence = torch.tensor([0.220833, 0.134783, 0.144348])
    
    # Run forward pass 
    hidden, out = manual.forward_pass(sequence)
    target = torch.tensor([0.189091])
    dy = (out[-1] - target).view(1,1)
    dWy, dB, dWh, dWx, dG = manual.backward_pass(sequence, hidden, dy)

    auto.zero_grad()
    hidden_auto, outputs_auto = auto(sequence)
    y_last = outputs_auto[-1] 
    loss = 0.5 * (y_last - target).pow(2).sum()
    loss.backward()
    
    check_close("InputWeights", dWx, auto.Wx.grad)
    print("---------------------------------------")
    check_close("HiddenWeights", dWh, auto.Wh.grad)
    print("---------------------------------------")
    check_close("OutputWeights", dWy, auto.Wy.grad)
    print("---------------------------------------")
    check_close("HiddenBias", dG, auto.bh.grad)
    print("---------------------------------------")
    check_close("OutputBias", dB, auto.by.grad)
    return 

def main():
    test_manual_grad()
    return 1

if __name__ == "__main__":
    main() 