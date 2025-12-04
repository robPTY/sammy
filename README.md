# sammy

Implementation of Meta's SAM, from scratch.

## Roadmap

### RNN

Full implementation of an RNN, using it as the starting point for the LSTM. The LSTM will serve as the transition towards learning Transformers, and then moving onto Vision Transformers (ViTs)

- [x] Forward pass
- [x] Backward pass
- [ ] Testing (manual gradients vs. autograd, loss vs. validation set)
- [ ] Optimization of network (learning rate decay, expanding onto more layers, not pre-defining sequence length)
