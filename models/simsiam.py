import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50

import torch.distributed as dist

def D(p, z, random_matrix=None, version='simplified'): 
    if version == 'original':
        z = z.detach()            # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    elif version == 'random':
        p = torch.matmul(p, random_matrix)
        z = torch.matmul(z, random_matrix)
        return - F.cosine_similarity(p, z.detach() , dim=-1).mean()

    else:
        raise Exception


def Cal_Loss_Matrix(z1, z2, random_matrix, margin=1.0):
    Triplet_loss = torch.tensor(0.).cuda()
    Triplet_loss.requires_grad = True
    N, Z = z1.shape 
    device = z1.device 

    representations = torch.cat([z1, z2], dim=0)               # 未归一化
    representations_norm = F.normalize(representations, dim=1) # 归一化
    representations_temp = torch.matmul(representations, random_matrix)                             # 2N x Z
    representations_temp = F.normalize(representations_temp, dim=1)

    similarity_matrix = torch.matmul(representations_temp, torch.transpose(representations_norm, 0, 1))  # 2N x 2N
    l_pos = torch.diag(similarity_matrix, N)                                                        # 得到 1~N+1 ... 共N个正样本对
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)                                            # 2N x 1
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]
    negatives = similarity_matrix[~diag].view(2*N, -1)                                              # 2N x 2(N-1)
    temp_loss = margin + negatives - positives
    temp_loss = torch.clamp(temp_loss, min=0.)
    Triplet_loss = torch.mean(temp_loss)
    return Triplet_loss


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimSiam(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( 
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()
    
    def forward(self, x1, x2):

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}

class SimSiam_ROMA(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( 
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()
        
        self.random_matrix = None
        self.random_in_dim = 2048
        self.random_out_dim = 2048
        
    def change_random_matrix(self, device):
        random_matrix = torch.randn(self.random_in_dim, self.random_out_dim).cuda()
        if dist.is_initialized():
            dist.broadcast(random_matrix, src=0)
        self.random_matrix = random_matrix
        if dist.is_initialized() and dist.get_rank() == 0:
            print('change random matrix')
    
    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2, self.random_matrix, version='random') / 2 + D(p2, z1, self.random_matrix, version='random') / 2
        return {'loss': L}
