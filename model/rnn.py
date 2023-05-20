import torch
import torch.nn as nn

class RNNModel(nn.Module):
    '''
    모델 구조
    input -> embedding -> rnn -> classifier

    cfg를 이용하여 각 layer의 파라미터를 바꿀 수 있다.
    '''
    def __init__(self, out_dim, cfg):
        super(RNNModel,self).__init__()
        self.out_dim = out_dim
        self.in_dim = cfg.vs
        self.emb_dim = cfg.emb
        self.hidden_dim = cfg.hidden
        self.emb_layer = nn.Embedding(num_embeddings = self.in_dim, embedding_dim = self.emb_dim, padding_idx = 0)
        self.rnn = nn.RNN(input_size = self.emb_dim, hidden_size = self.hidden_dim, batch_first = True)
        self.classifier = nn.Linear(self.hidden_dim, self.out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        input x : batch x time step 
        embedded : batch x time step x emb_dim
        output : batch x time step x hidden_dim
        return : batch x out_dim
        """
        embedded = self.emb_layer(x)
        output, _ = self.rnn(embedded)
        #가장 마지막 time step만 사용해주기 때문에 [:,-1,:]로 slicing 한다
        return self.sigmoid(self.classifier(output[:,-1,:]))