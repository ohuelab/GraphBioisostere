# config.py

class Args:
    def __init__(self):
        # Data
        # self.data_dir = "data/graph_batches_sub"
        self.batch_size = 4096
        self.seed = 41

        # Model
        self.node_in = 30
        self.edge_in = 11
        self.hidden_dim = 64
        self.embedding_dim = 64
        self.num_layers = 2
        self.dropout = 0.2

        # Training
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.epochs = 500
        self.patience = 30

        # Loss
        self.loss_type = "pair"  # "single", "pair", "all"
        # self.loss_weights = {"pred_ab": 0.5, "pred_a": 0.25, "pred_b": 0.25}

args = Args()
