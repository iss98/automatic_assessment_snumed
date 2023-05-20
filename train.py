from tqdm import  tqdm
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import BinaryAccuracy


from cfg import get_cfg
from tok import get_tokenizer
from preprocess import get_ds
from model import get_model
 

if __name__ == "__main__":
    cfg = get_cfg()
    #모델 이름 설정 : 문항_모델이름_토크나이저_vocab size ex) 1-1_rnn_tokenizer_40
    model_name = f"{cfg.item}_{cfg.model}_{cfg.tokenizer}_{cfg.vs}"
    #wandb에 연결하기
    wandb.init(project=cfg.prjname, name = model_name, config=cfg)
    #tokenizer 불러오기
    tokk = get_tokenizer(cfg)(model_name, cfg)
    #ds 불러오기
    ds = get_ds(cfg)(tokk, cfg)

    #데이터 로드 이후 프린트하기
    print("================================================")
    print("데이터 로드 성공")
    print(f"데이터의 크기(학생들 풀이의 개수) : {len(ds)}")
    print("모델 정보 : " + model_name)
    print("================================================")

    #train, test나누어서 ds 로드하기
    #데이터셋 크기가 작아서 valid set은 일단 주석 처리
    test_len = round(len(ds)*cfg.testsplit)
    # val_len = round((len(ds) - test_len) * cfg.valsplit)

    # 데이터셋을 섞어서 train과 test 비율대로 나누기
    train_ds, test_ds = random_split(ds, [len(ds)-test_len, test_len])
    # train_ds, val_ds = random_split(tv_ds, [len(ds)-test_len-val_len, val_len])

    # 학습에서 활용할 수 있게 dataloader 설정
    train_dl = DataLoader(train_ds, batch_size = cfg.bs, shuffle=True)
    # val_dl = DataLoader(val_ds, batch_size = cfg.bs, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size = cfg.bs, shuffle=True)

    # model이 두 가지 나뉘기 때문에 output을 지정해주어야한다. 데이터셋에 저장해놓음
    model = get_model(cfg)(ds.output_d, cfg)
    # 학습을 위해 그래픽카드로 옮기기
    model.to(cfg.device)

    #Stochastic gradient descent를 이용하여 학습함
    optimizer = optim.SGD(model.parameters(), lr = cfg.lr, momentum = cfg.mt)
    
    loss_fn = nn.BCELoss().to(cfg.device)
    ACC_fn = BinaryAccuracy(threshold = 0.5)
    
    test_every = cfg.testevery

    # 가장 좋은 모델 설정을 위한 방법
    best_acc = 0
    # 모델 저장할 위치 설정
    fname = "save/"+model_name+".pt"
    print("================================================")
    print("모델설정 완료 학습 시작")
    print("파일 저장 위치 : " + fname)
    print("================================================")
    
    # 학습
    for ep in range(cfg.epoch):
        model.train()
        loss_ep = []
        acc_ep = []
        #train
        for batch in tqdm(train_dl):
            
            response = batch["input"].to(cfg.device)
            target = batch["label"].to(cfg.device)

            optimizer.zero_grad()
            predict = model(response)

            loss = loss_fn(predict.to(torch.float32), target.to(torch.float32))

            loss.backward()
            optimizer.step()
            #loss 값 계산하고 리스트에 넣어주기
            loss_ep.append(loss.item())

            #acc 값 계산하고 리스트에 넣어주기
            acc_ep.append(ACC_fn(predict.detach().cpu(), target.detach().cpu()))
            
        if (ep + 1) % test_every == 0:
            model.eval()
            t_loss = []
            t_acc = []
            with torch.no_grad():
                for batch in test_dl:                 
                    response = batch["input"].to(cfg.device)
                    target = batch["label"].to(cfg.device)
                    predict = model(response)
                    loss = loss_fn(predict.to(torch.float32), target.to(torch.float32))
                    #loss 값 계산하고 리스트에 넣어주기
                    t_loss.append(loss.item())

                    #acc 값 계산하고 리스트에 넣어주기
                    t_acc.append(ACC_fn(predict.detach().cpu(), target.detach().cpu()))

            wandb.log({"test_acc": np.mean(t_acc)},commit=False)
            if np.mean(t_acc) > best_acc :
                wandb.run.summary["best_acc"] = np.mean(t_acc)
                best_auc = np.mean(t_acc)
                torch.save(model.state_dict(), fname)
        
        #이번 epoch의 train data 넣어주기
        wandb.log({"loss": np.mean(loss_ep), "acc": np.mean(acc_ep), "ep": ep})