# config.py

class Config(object):
    embed_size = 200
    num_channels = 32
    kernel_size = [2,3,4,5]
    output_size = 5
    max_epochs = 50
    lr = 1e-3
    batch_size = 32
    max_sen_len = 128
    dropout_keep = 0.3