from .rds import ResponseDataset
from .info_rds import InfoResponseDataset
from .m_rds import MCResponseDataset

def get_ds(cfg):
    if cfg.concept == "kc":
        if cfg.item == "all":
            return InfoResponseDataset
        else : return ResponseDataset
    elif cfg.concept == "mc":
        return MCResponseDataset