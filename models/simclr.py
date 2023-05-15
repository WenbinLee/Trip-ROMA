import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

import torch.distributed as dist

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
    
def batch_all_gather(x):
    if not dist.is_available() or not dist.is_initialized():
        return x
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)

def NT_XentLoss(z1, z2, temperature=0.5, random_matrix=None):
    z1 = batch_all_gather(z1)
    z2 = batch_all_gather(z2)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    if random_matrix is not None:
        representations = torch.matmul(representations, random_matrix)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]
    negatives = similarity_matrix[~diag].view(2*N, -1)
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature
    labels = torch.zeros(2*N, device=device, dtype=torch.int64)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer3(x)
        return x 

class SimCLR(nn.Module):

    def __init__(self, backbone=resnet50()):
        super().__init__()
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        loss = NT_XentLoss(z1, z2)
        return {'loss':loss}


class SimCLR_ROMA(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()
        self.backbone = backbone
        out_dim = 128
        self.projector = projection_MLP(backbone.output_dim, out_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        
        self.random_matrix = None
        self.random_in_dim = out_dim
        self.random_out_dim = out_dim
        
    def change_random_matrix(self, device):
        random_matrix = torch.randn(self.random_in_dim, self.random_out_dim).to(device)
        if dist.is_initialized():
            dist.broadcast(random_matrix, 0)
        self.random_matrix = random_matrix
        if dist.is_initialized() and dist.get_rank() == 0:
            print('change random matrix')

    def forward(self, input1, input2):
        z1 = self.encoder(input1)
        z2 = self.encoder(input2)
        loss = NT_XentLoss(z1, z2, self.random_matrix)
        return {'loss':loss}
