# config.py

class Config(object):
    embed_size = 300
    num_channels = 100
    kernel_size = [3,4,5]
    output_size = 5
    max_epochs = 15
    lr = 0.3
    batch_size = 64
    max_sen_len = 128
    dropout_keep = 0.8