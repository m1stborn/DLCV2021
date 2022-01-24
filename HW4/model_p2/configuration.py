class Config(object):
    def __init__(self):

        self.lr = 3e-4

        # Epochs
        self.epochs = 100
        self.start_epoch = 0

        # Basic
        self.n_worker = 4
        self.batch_size = 64


class ConfigResnet(object):
    def __init__(self):

        self.lr = 1e-3

        # Epochs
        self.epochs = 40
        self.start_epoch = 0

        # Basic
        self.n_worker = 4
        self.batch_size = 64
