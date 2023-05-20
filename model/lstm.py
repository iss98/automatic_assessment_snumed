import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    '''
    모델 구조
    input -> embedding -> lstm -> classifier

    cfg를 이용하여 각 layer의 파라미터를 바꿀 수 있다.
    '''
    def __init__(self, out_dim, cfg):
        super(LSTMModel,self).__init__()
        self.out_dim = out_dim
        self.in_dim = cfg.vs
        self.emb_dim = cfg.emb
        self.hidden_dim = cfg.hidden
        self.device = cfg.device

        self.emb_layer = nn.Embedding(num_embeddings = self.in_dim, embedding_dim = self.emb_dim, padding_idx = 0)
        self.lstm = nn.LSTM(input_size = self.emb_dim, hidden_size = self.hidden_dim, batch_first = True)
        self.classifier = nn.Linear(self.hidden_dim, self.out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        input x : batch x time step 
        embedded : batch x time step x emb_dim
        output : batch x time step x hidden_dim
        return : batch x out_dim
        """
        h0 = torch.zeros(x.size(0), 1, self.hiddem_dim).to(self.device)
        c0 = torch.zeros(x.size(0), 1, self.hiddem_dim).to(self.device)
        embedded = self.emb_layer(x)
        output, _ = self.lstm(embedded, (h0, c0))
        #가장 마지막 time step만 사용해주기 때문에 [:,-1,:]로 slicing 한다
        return self.sigmoid(self.classifier(output[:,-1,:]))