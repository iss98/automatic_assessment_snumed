from .rnn import RNNModel
from .lstm import LSTMModel
from .att import ATTModel

model_list = ["rnn", "lstm", "att"]

def get_model(cfg):
    if cfg.model == "rnn":
        return RNNModel
    elif cfg.model == "lstm":
        return LSTMModel
    elif cfg.model =="att":
        return ATTModel
