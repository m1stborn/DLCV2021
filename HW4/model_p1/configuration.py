class Config(object):
    def __init__(self):

        # Learning Rates
        self.lr = 1e-3
        self.seed = 1

        # Epochs
        self.epochs = 200
        self.start_epoch = 0

        # Basic
        self.device = 'cuda'
        self.n_worker = 4
        # self.batch_size = 32
        # self.checkpoint = './checkpoint.pth'

        # Model
        self.n_batch = 100
        self.n_way = 30
        self.n_shot = 1
        self.n_query = 15
        # batch_size = N_way * (N_query + N_shot) = 30 * (15 + 1) = 480

        # Valid
        self.val_n_way = 5
        self.val_n_shot = 1
        self.val_n_query = 15
