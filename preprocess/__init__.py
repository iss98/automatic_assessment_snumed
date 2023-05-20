from .rds import ResponseDataset
from .info_rds import InfoResponseDataset

def get_ds(cfg):
    if cfg.item == "all":
        return InfoResponseDataset
    else : return ResponseDataset