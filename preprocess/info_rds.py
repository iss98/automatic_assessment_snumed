'''
이 데이터셋은 전체 문항의 채점 모델을 생성할 때 사용된다.
한 학생에 대한 데이터는 다음과 같이 표현된다

(x y)
x 는 문제 + 학생응답으로 구성
y 정오답

응답의 최대 길이는 cfg.ml로 결정되어 있음
'''
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from pathlib import Path
import os

class InfoResponseDataset(Dataset):
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.path = [str(x) for x in Path("./data/response").glob("*.csv")] #전체 데이터셋 path 가져오기
        self.data = [] #리스트 안에 튜플 구조로 데이터를 넣는다 (학생들의 풀이, 지식요소)
                        #헉샹둘의 풀이는 인코딩해서 넣는다
        self.item_info_preprocess()
        self.max_len = cfg.ml #padding을 위해 풀이의 최대 길이를 설정하기
        self.output_d = 1
        self.encode_data(cfg)
    
    def encode_data(self, cfg):
        if cfg.tokenizer == "sp":
            for path in tqdm(self.path):
                df = pd.read_csv(path)
                df = df.dropna(subset = ["답안"])
                item_name = os.path.basename(path) #문제의 질문 가지고오기
                item_name = os.path.splitext(item_name)[0]
                info = self.item_info[item_name]
                for row in df:
                    x = "문제 " + info + " 풀이 " + row[1]
                    token = self.tokenizer.tokenizer(x)
                    y = [int(row.loc["정오답"])]
                    self.data.append((token['input_ids'],y))
        elif cfg.tokenizer == "bw":
            for path in tqdm(self.path):
                df = pd.read_csv(path)
                df = df.dropna(subset = ["답안"])
                item_name = os.path.basename(path) #문제의 질문 가지고오기
                item_name = os.path.splitext(item_name)[0]
                info = self.item_info[item_name]
                for row in df:
                    x = "문제 " + info + " 풀이 " + row[1]
                token = self.tokenizer.tokenizer(x)
                y = [int(row.loc["정오답"])]
                self.data.append((token.ids,y))

    def item_info_preprocess(self):
        df = pd.read_csv("./data/info/info.csv")
        self.item_info = {}
        for row in df:
            self.item_info[row[0]] = row[1]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        # padding 해주면서 데이터를 어떻게 가지고 올지 정의하기
        x, y = self.data[idx]
        if len(x) < self.max_len :
            pad_x = [0] * (self.max_len - len(x)) + x
        elif len(x) >= self.max_len:
            diff = len(x) - self.max_len
            pad_x = x[diff:]
        return dict(
            input = torch.tensor(pad_x, dtype = torch.long),
            label = torch.tensor(y, dtype = torch.long)
        )
