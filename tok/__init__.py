from .sptokenizer import SPTokenizer
from .bwtokenizer import BWTokenizer

tokenizer_list = ["sp", "bw"]

def get_tokenizer(cfg):
    if cfg.tokenizer == "sp" :
        return SPTokenizer
    elif cfg.tokenizer == "bw":
        return BWTokenizer