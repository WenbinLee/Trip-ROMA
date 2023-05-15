from tqdm import tqdm
import torch.nn.functional as F 
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def knn_monitor_ddp(net, memory_data_loader, test_data_loader, epoch, k=200, t=0.1, hide_progress=False, device='cpu', local_rank=-1):
    net.eval()
    
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        if dist.get_rank() == 0:
            local_progress = tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress)
        else:
            local_progress = memory_data_loader
        for data, target in local_progress:
            # do not need epoch seed for DDP in KNN
            feature = net(data.to(device, non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        if dist.get_rank() == 0:
            test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
        else:
            test_bar = test_data_loader
        for data, target in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            if dist.get_rank() == 0:
                test_bar.set_postfix({'Accuracy':total_top1 / total_num * 100})
    return total_top1 / total_num * 100

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels