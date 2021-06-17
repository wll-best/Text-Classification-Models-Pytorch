# BiGRU Model
Here, we have implemented a Bi-directional Gated Ｒecurrent Unit network in PyTorch.

## Model Architecture
The architecture of Bi-directional GRU is as follows:



## Implementation Details
- word2vec Embeddings for initializing word vectors
- 2 Layers of BiLSTM
- Used  hidden units within each BiGRU layer
- Dropout with keep probability 0.3
- Optimizer - Adam
- Loss function - CrossEntropyLoss
- Experimented with flexible sequence lengths and sequences of length 128
