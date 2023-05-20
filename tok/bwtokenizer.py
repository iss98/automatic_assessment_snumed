from tokenizers import BertWordPieceTokenizer
from pathlib import Path

class BWTokenizer():
    def __init__(self, model_name, cfg):
        self.model_name = model_name
        self.data_dir = "./data/text_data" #텍스트만 따로 추출한 데이터
        self.file = self.make_file(cfg) #훈련시킬 파일들 리스트 만들기
        self.tokenizer = BertWordPieceTokenizer(lowercase = cfg.lc , strip_accents = cfg.sa)
        self.tokenizer.train(files = self.file,
                             vocab_size = cfg.vs,
                             limit_alphabet = cfg.la,
                             min_frequency = cfg.mf)
        self.tokenizer.save_model("./tok/save", self.model_name)

    def make_file(self, cfg):
        # 모든 문제에 대한 모델을 만드는 경우, 특정 문제에 대한 모델을 만드는 경우를 나누기 위해 파일을 어떻게 가지고 올지 나눔
        if cfg.item == "all" :
            paths = [str(x) for x in Path(self.data_dir).glob("*.txt")]
        else :
            paths = [str(x) for x in Path(self.data_dir).glob(cfg.item+".txt")]
        return paths