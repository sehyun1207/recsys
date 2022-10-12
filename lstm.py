import torch
import torch.nn as nn
from torch.autograd import Variable
import  torch.nn.functional as F

def flatten(t):
    return [item for sublist in t for item in sublist]

class RNN2(nn.Module):
    def __init__(self,  tr_matrices, tr_categoricals, tr_scalar,embedding_dim, hidden_dim, num_items, num_output, num_feats=903):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.feat_embeddings = nn.Embedding(num_feats, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim*7, hidden_dim,batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_items)
        self.item_mat_key = ['item_last_20', 'item_last_5', 'item_last_2', 'item_last_1']
        self.feat_mat_key = ['item_last_5_feats', 'item_last_3_feats', 'item_last_1_feats']
        dropout = 0.1
        self.drop = nn.Dropout(dropout)

        modules = {}
        cat_modules = {}
        bns = {}
        cat_bns = {}
        aggr_width = 0


        for mat_id, mat in tr_matrices.items():
            if mat_id not in self.feat_mat_key + self.item_mat_key:
                continue 
            print(mat_id)
            width = mat.shape[-1]
            modules[mat_id] = nn.ModuleList([nn.Linear(width, embedding_dim)])
            bns[mat_id] = nn.ModuleList([nn.BatchNorm1d(embedding_dim)])
            aggr_width += embedding_dim


        self.cat_embs = nn.ModuleDict(cat_modules)
        self.all_bns = nn.ModuleDict(bns)
        self.cat_bns = nn.ModuleDict(cat_bns)

    def init_hidden(self,x):
        return (Variable(torch.zeros(1, x, self.hidden_dim).to('cuda:0')),
                Variable(torch.zeros(1, x, self.hidden_dim).to('cuda:0')))

    def forward(self, mats, cats, scalar_feats, norm_user=False):
        self.hidden = self.init_hidden(scalar_feats.size(0))
        item_embs = self.item_embeddings.weight.data.normal_(0, 0.001)
        feat_embs = self.feat_embeddings.weight.data.normal_(0, 0.001)

        aggr = []
        for mat_key in self.item_mat_key:
            h = mats[mat_key]
            h = self.all_bns[mat_key][0](self.drop(F.linear(h, item_embs.T)))
            aggr.append(h)

        for mat_key in self.feat_mat_key:
            h = mats[mat_key]
            h = self.all_bns[mat_key][0](self.drop(F.linear(h, feat_embs.T)))
            aggr.append(h)

        h = torch.cat(aggr, dim=-1)
        output, self.hidden = self.lstm(h.view(h.size()[0],1,-1), self.hidden)
        rating_scores = self.linear(output.view(len(h), -1))
        return rating_scores
