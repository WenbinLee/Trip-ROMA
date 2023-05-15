from random import shuffle
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from arguments_ddp import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def main(args, device, local_rank):
    
    train_set = get_dataset(transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs), train=True, train_classifier=True, **args.dataset_kwargs)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train.batch_size // dist.get_world_size(), sampler=train_sampler, **args.dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs
    )

    assert args.eval_from is not None  
    model = get_backbone(args.model.backbone)
    classifier = nn.Linear(in_features=model.output_dim, out_features=1000, bias=True).to(args.device)
    
    save_dict = torch.load(args.eval_from, map_location='cpu')
    msg = model.load_state_dict({k[9:]:v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}, strict=True)

    # print(msg)
    model = model.to(args.device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    classifier = DDP(classifier, device_ids=[local_rank], output_device=local_rank)
    optimizer = get_optimizer(
        args.eval.optimizer.name, classifier,
        lr=args.eval.base_lr*args.eval.batch_size/256,
        momentum=args.eval.optimizer.momentum,
        weight_decay=args.eval.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs, args.eval.warmup_lr*args.eval.batch_size/256,
        args.eval.num_epochs, args.eval.base_lr*args.eval.batch_size/256, args.eval.final_lr*args.eval.batch_size/256,
        len(train_loader),
    )

    loss_meter = AverageMeter(name='Loss')
    acc_meter = AverageMeter(name='Accuracy')

    # Start training
    if dist.get_rank() == 0:
        global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
    else:
        global_progress = range(0, args.eval.num_epochs)
    for epoch in global_progress:
        loss_meter.reset()
        model.eval()
        classifier.train()
        
        if dist.get_rank() == 0:
            local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)
        else:
            local_progress = train_loader
        
        for idx, (images, labels) in enumerate(local_progress):
            train_loader.sampler.set_epoch(epoch)

            classifier.zero_grad()
            with torch.no_grad():
                feature = model(images.to(args.device))
                # print(feature.size())

            preds = classifier(feature)

            loss = F.cross_entropy(preds, labels.to(args.device))

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            lr = lr_scheduler.step()
            if dist.get_rank() == 0:
                local_progress.set_postfix({'lr':lr, "loss":loss_meter.val, 'loss_avg':loss_meter.avg})

    classifier.eval()
    correct, total = 0, 0
    acc_meter.reset()
    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            feature = model(images.to(args.device))
            preds = classifier(feature).argmax(dim=1)
            correct = (preds == labels.to(args.device)).sum().item()
            acc_meter.update(correct/preds.shape[0])
    if dist.get_rank() == 0:
        print(f'Accuracy = {acc_meter.avg*100:.2f}')
    
    # FIXME 加入保存linear的结果
    # path = args.eval_from
    # path = os.path.join(*(path.split('/')[:-1])), 'linear-test.txt')
    # with open(path, 'w') as f:
    #     text = time.time().__str__() + ":" + f'Accuracy = {acc_meter.avg*100:.2f}'
    #     f.write(text)


if __name__ == "__main__":
    args, device, local_rank = get_args(create_log=False)
    main(args=args, device=device, local_rank=local_rank)