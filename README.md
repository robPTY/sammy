# sammy

Implementation of Meta's SAM, from scratch.

## Roadmap

### RNN

Full implementation of an RNN, using it as the starting point for the LSTM. The LSTM will serve as the transition towards learning Transformers, and then moving onto Vision Transformers (ViTs)

- [x] Forward pass
- [x] Backward pass
- [x] Testing (manual gradients vs. autograd, loss vs. validation set)
- [ ] Optimization of network (learning rate decay, expanding onto more layers, not pre-defining sequence length)

### LSTM

Implemented 1997 Hochreiter LSTM to predict a an n-length sequence of sun spots given an n-length input sequence. Since the results are hard to visualize, I also will implement Seq2Seq for translation between English and Spanish (since I can properly verify this).

- [x] Forward pass
- [x] Backward pass
- [x] Testing (manual gradients vs. autograd, training loss vs. validation loss)
- [x] Seq2Seq 2014 Paper Implementation (corpora size of 142,928 words)

### Transformer

Full implementation of 2017 Vaswani et al paper. Using this as the starting point for sammy, the ViT.

- [ ] Tokenizer
- [ ] Attention
- [ ] Forward Pass
- [ ] Backward Pass

## References

Across this project, I've probably used countless resources, but the most important ones so far

##### ML Concepts

- [2003 Bengio et. al](https://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [CS231n Backpropagation Lecture](https://www.youtube.com/watch?v=i94OvYb6noo&list=WL&index=1)
- [1997 Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [2014 Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215)
- [Let's Build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
