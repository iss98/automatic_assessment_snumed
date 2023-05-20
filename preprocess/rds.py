'''
이 데이터셋은 문항 1개만을 위한 모델을 만들 때 사용된다.
한 학생에 대한 데이터는 다음과 같이 표현된다

(x,y)
x 는 학생들의 응답을 인코딩한 결과
y 는 지식요소에 대한 라벨링

응답의 최대 길이는 cfg.ml로 결정되어 있음
'''
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class ResponseDataset(Dataset):
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.path = "./data/response/"+cfg.item+".csv" #훈련시킬 파일 패스 가지고오기
        self.data = [] #리스트 안에 튜플 구조로 데이터를 넣는다 (학생들의 풀이, 지식요소)
                        #헉샹둘의 풀이는 인코딩해서 넣는다
        self.max_len = cfg.ml #padding을 위해 풀이의 최대 길이를 설정하기
        self.encode_data(cfg)
    
    def encode_data(self, cfg):
        df = pd.read_csv(self.path)
        df = df.dropna(subset = ["답안"]) #답안이 없는 학생 제거
        column_index = df.columns.get_loc("정오답") #각 문항마다 지식요소가 다르기 때문에
        self.output_d = column_index - 2
        if column_index == 3 :
            if cfg.tokenizer == "sp":
                for _, row in tqdm(df.iterrows()):
                    x = row[1]
                    token = self.tokenizer.tokenizer(x)
                    y = [row[2]]
                    self.data.append((token['input_ids'],y))
            elif cfg.tokenizer == "bw":
                for _, row in tqdm(df.iterrows()):
                    x = row[1]
                    token = self.tokenizer.tokenizer.encode(x)
                    y = [row[2]]
                    self.data.append((token.ids,y))
        else :
            if cfg.tokenizer == "sp":
                for _, row in tqdm(df.iterrows()):
                    x = row[1]
                    token = self.tokenizer.tokenizer(x)
                    y = [a for a in row[2:column_index]]
                    self.data.append((token['input_ids'],y))
            elif cfg.tokenizer == "bw":
                for _, row in tqdm(df.iterrows()):
                    x = row[1]
                    token = self.tokenizer.tokenizer.encode(x)
                    y = [a for a in row[2:column_index]]
                    self.data.append((token.ids,y))

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
