from math import gamma
import torch
from torch import random
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.models import resnet50


def Cal_Loss_Matrix(z1, z2, random_matrix=None, margin=1):
        N, Z = z1.shape      
        device = z1.device 
        representations = torch.cat([z1, z2], dim=0)
        if random_matrix is not None:
            representations = torch.matmul(representations, random_matrix)
        representations = F.normalize(representations, dim=1)
        similarity_matrix = torch.matmul(representations, torch.transpose(representations, 0, 1))
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        diag = torch.eye(2*N, dtype=torch.bool, device=device)
        diag[N:,:N] = diag[:N,N:] = diag[:N,:N]
        negatives = similarity_matrix[~diag].view(2*N, -1)

        temperature = 0.5
        negative_ce = negatives.permute(1, 0).reshape(-1, 1)
        positive_ce = positives.repeat([2 * N - 2, 1])
        
        logits = torch.cat([positive_ce,negative_ce], dim=-1)
        labels = torch.zeros(2 * N * (2*N - 2), device=device, dtype=torch.int64)
        logits /= temperature
        CE_loss = F.cross_entropy(logits, labels, reduction='mean') 
        
        temp_loss = margin + negative_ce - positive_ce
        temp_loss = torch.clamp(temp_loss, min=0.)
        mask = torch.gt(temp_loss, 1e-16)
        Triplet_loss = torch.mean(temp_loss[mask])

        return (Triplet_loss + CE_loss * 8)


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_dim),
        )
        self.bnrelu = nn.Sequential(nn.BatchNorm1d(hidden_dim),
                                    nn.LeakyReLU(0.2, True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.LeakyReLU(0.2, True))
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim, out_dim),
                                    nn.BatchNorm1d(out_dim))
        # self.dropout = nn.Dropout(p=0.2)
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers
    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.bnrelu(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.bnrelu(x)
            x = self.layer3(x)
        elif self.num_layers == 1:
            x = self.layer3(x)
        elif self.num_layers == 0:
            x = x
        else:
            raise Exception
        return x


class Trip(nn.Module):
    def __init__(self, backbone=resnet50):
        super(Trip, self).__init__()

        self.backbone = backbone
        self.projector = projection_MLP(in_dim=backbone.output_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        self.margin = 1
        
    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        loss = Cal_Loss_Matrix(z1, z2, margin = self.margin)
        return {'loss':loss}


class Trip_ROMA(nn.Module):
    def __init__(self, backbone=resnet50):
        super(Trip_ROMA, self).__init__()

        self.backbone = backbone
        self.projector = projection_MLP(in_dim=backbone.output_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        self.margin = 1
        
        self.random_matrix = None
        self.random_in_dim = 2048
        self.random_out_dim = 2048
        
    def change_random_matrix(self, device):
        # 标准正态分布
        random_matrix = torch.randn(self.random_in_dim, self.random_out_dim).to(device)

        # 伯努利分布
        # bernoulli_seed = torch.empty(self.random_in_dim, self.random_out_dim).uniform_(0, 1)
        # random_matrix = torch.bernoulli(bernoulli_seed).to(device)

        # [-1, 1]均匀分布
        # random_matrix = torch.Tensor(self.random_in_dim, self.random_out_dim).uniform_(-1,1).to(device)
        
        if dist.is_initialized():
            dist.broadcast(random_matrix, 0)
        self.random_matrix = random_matrix
        if dist.is_initialized() and dist.get_rank() == 0:
            print('change random matrix')

    def forward(self, x1, x2):
        x1, x2 = self.encoder(x1), self.encoder(x2)
        loss = Cal_Loss_Matrix(x1, x2, self.random_matrix, margin = self.margin)
        return {'loss': loss}
            
    
            

        
        
