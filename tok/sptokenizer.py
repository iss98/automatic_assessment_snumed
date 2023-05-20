import os
import csv
import sentencepiece as spm
from pathlib import Path
import shutil
from transformers import T5Tokenizer


class SPTokenizer():
    def __init__(self, model_name, cfg):
        self.model_name = model_name #모델 이름
        self.data_dir = "./data/text_data" #텍스트만 따로 추출한 데이터
        self.corpus = self.make_corpus(cfg) #코퍼스 만들기
        spm.SentencePieceTrainer.train( f"--input={self.corpus} "+" --model_prefix="+self.model_name+f" --vocab_size={cfg.vs}" + 
                                       " --model_type=bpe" +
                                       " --max_sentence_length=999999" + # 문장 최대 길이 (너무 길면 에러발생)
                                       " --pad_id=0 --pad_piece=<pad>" + # pad (0)
                                       " --unk_id=1 --unk_piece=<unk>" + # unknown (1)
                                       " --bos_id=2 --bos_piece=<s>" + # begin of sequence (2)
                                       " --eos_id=3 --eos_piece=</s>" + # end of sequence (3)
                                       " --user_defined_symbols=<sep>,<cls>,<mask>") # 사용자 정의 토큰
        #만든 파일들은 token/save에 저장하기
        shutil.move(self.model_name+".vocab", "./tok/save/"+self.model_name+".vocab")
        shutil.move(self.model_name+".model", "./tok/save/"+self.model_name+".model")
        #tokenizer 만들기
        self.tokenizer = T5Tokenizer(vocab_file="./tok/save/"+self.model_name+".model")
        self.tokenizer.save_pretrained("./tok/save/"+self.model_name)

    def make_corpus(self, cfg):
        # 모든 문제에 대한 모델을 만드는 경우, 특정 문제에 대한 모델을 만드는 경우를 나누기 위해 파일을 어떻게 가지고 올지 나눔
        if cfg.item == "all" :
            paths = [str(x) for x in Path(self.data_dir).glob("*.txt")]
        else :
            paths = [str(x) for x in Path(self.data_dir).glob(cfg.item+".txt")]
        return ",".join(paths)
    
