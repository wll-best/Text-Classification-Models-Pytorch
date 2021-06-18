# config.py

class Config(object):
    embed_size = 300
    num_channels = 32
    kernel_size = [2,3,4,5]
    output_size = 5
    max_epochs = 15
    lr = 1e-1
    batch_size = 32
    max_sen_len = 128
    dropout_keep = 0.7