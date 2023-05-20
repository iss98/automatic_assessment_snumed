from .rnn import RNNModel

model_list = ["rnn", "lstm", "att"]

def get_model(cfg):
    if cfg.model == "rnn":
        return RNNModel