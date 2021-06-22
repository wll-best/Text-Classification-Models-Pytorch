# config.py

class Config(object):
    embed_size = 200#w2v维度
    pos_embed_size =20#pos维度
    hidden_layers = 2
    hidden_size = 300
    bidirectional = True
    output_size = 5
    max_epochs = 3
    lr = 0.001
    batch_size = 64
    max_sen_len = 128 # Sequence length for BiGRU
    dropout_keep = 0.3