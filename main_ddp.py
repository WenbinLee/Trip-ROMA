import os
from threading import local
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments_ddp import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, knn_monitor_ddp, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from linear_eval_ddp import main as linear_eval_ddp
from datetime import datetime
import time
import torch.distributed as dist
import math
from torch.nn.parallel import DistributedDataParallel as DDP
torch.manual_seed(1)

def main(args, device, local_rank):
    torch.backends.cudnn.benchmark = True
    train_set = get_dataset(transform=get_aug(train=True, **args.aug_kwargs), train=True, **args.dataset_kwargs)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=False, batch_size=args.train.batch_size // dist.get_world_size(), sampler=train_sampler, **args.dataloader_kwargs)
    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # define model
    model = get_model(args.model)
    if dist.get_rank() == 0:
        print(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )
    
    if dist.get_rank() == 0:
        logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0 
    # Start training
    if dist.get_rank() == 0:
        global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    else:
        global_progress = range(0, args.train.stop_at_epoch)
    for epoch in global_progress:
        train_loader.sampler.set_epoch(epoch)

        model.train()
        
        if dist.get_rank() == 0:
            local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        else:
            local_progress= train_loader
            
        # same randommatrix each 10 epoch
        if (epoch == 0 or (epoch + 1) % args.train.random_matrix_epoch_interval == 0) and hasattr(model.module, 'change_random_matrix'):
            model.module.change_random_matrix(device=device)
        
        for idx, ((images1, images2), labels) in enumerate(local_progress):
            model.zero_grad()
            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            # data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            loss = data_dict['loss'] # ddp
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict['loss'] = loss.item()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            
            if dist.get_rank() == 0:
                local_progress.set_postfix(data_dict)
                logger.update_scalers(data_dict)
            
        if dist.get_rank() == 0:
            epoch_dict = {"accuracy":accuracy}
            global_progress.set_postfix(epoch_dict)
            logger.update_scalers(epoch_dict)
    
            if hasattr(args.eval, 'linear_interval') and epoch % args.eval.linear_interval == 0 and epoch != 0:
                # Save checkpoint
                model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
                torch.save({
                    'epoch': epoch+1,
                    'state_dict':model.module.state_dict(),
                    'optimizer_dict':optimizer.state_dict()
                }, model_path)
                print(f"Model saved to {model_path}")
                with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
                    f.write(f'{epoch+1} epoch pth: {model_path}')
        
        torch.cuda.synchronize()
    
    # Save checkpoint
    model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
    if dist.get_rank() == 0:
        torch.save({
            'epoch': epoch+1,
            'state_dict':model.module.state_dict(),
            'optimizer_dict':optimizer.state_dict()
        }, model_path)
        print(f"Model saved to {model_path}")
        with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
            f.write(f'final pth: {model_path}')
            
    torch.cuda.synchronize()


if __name__ == "__main__":
    args, device, local_rank = get_args()

    main(args=args, device=device, local_rank=local_rank)
    
    if dist.get_rank() == 0:
        completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
        os.rename(args.log_dir, completed_log_dir)
        print(f'Log file has been saved to {completed_log_dir}')