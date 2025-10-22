import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import torch.nn as nn
from const import *

def custom_collate_fn(batch):
    """
    Custom collate function to handle batch data
    Args:
        batch: list of tuples from Dataset.__getitem__
    Returns:
        batched data as tensors
    """
    # Unzip the batch
    paths, images, targets = zip(*batch)
    
    # Stack images and targets into tensors
    images = torch.stack(images, 0)
    targets = torch.stack(targets, 0)
    
    return paths, images, targets


LOGGING_NAME="custom"
def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def poly_lr_scheduler(args, optimizer, epoch, power=2):
    lr = round(args.lr * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr



def train(args, train_loader, model, criterion, optimizer, epoch):
    model.train()

    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    LOGGER.info(('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss'))
    pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_,input, target) in pbar:
        if args.onGPU == True:
            input = input.cuda().float() / 255.0        
        output = model(input)
        
        # target=target.cuda()
        optimizer.zero_grad()

        focal_loss,tversky_loss,loss = criterion(output,target)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(('%13s' * 1 + '%13.4g' * 3) %
                                     (f'{epoch}/{300 - 1}', tversky_loss, focal_loss, loss.item()))
        

def train16fp(args, train_loader, model, criterion, optimizer, epoch,scaler):
    model.train()
    print("16fp-------------------")
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    LOGGER.info(('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss'))
    pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_,input, target) in pbar:
        optimizer.zero_grad()
        if args.onGPU == True:
            input = input.cuda().float() / 255.0        
        output = model(input)
        with torch.cuda.amp.autocast():
            focal_loss,tversky_loss,loss = criterion(output,target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.set_description(('%13s' * 1 + '%13.4g' * 3) %
                                     (f'{epoch}/{300 - 1}', tversky_loss, focal_loss, loss.item()))


import torch
from tqdm import tqdm
import numpy as np
from IOUEval import SegmentationMetric

def val(valLoader, model, device):
    '''
    Validation function
    :param valLoader: validation data loader
    :param model: model to validate
    :param device: device to run validation on
    :return: validation metrics
    '''
    model.eval()
    da_segment_results = SegmentationMetric(2)
    ll_segment_results = SegmentationMetric(2)
    
    with torch.no_grad():
        for i, (_, input, target) in enumerate(tqdm(valLoader)):
            # Normalize input to [0, 1] range
            input = input.to(device).float() / 255.0
            target = target.to(device)
            
            # Forward pass
            da_predict, ll_predict = model(input)
            
            # Process predictions - argmax over class dimension
            da_predict = torch.argmax(da_predict, dim=1)
            ll_predict = torch.argmax(ll_predict, dim=1)
            
            # Extract targets from 4-channel tensor [batch, 4, H, W]
            # Channels 0-1: drivable area (background, foreground)
            # Channels 2-3: lane lines (background, foreground)
            # Take argmax to get class labels (0 or 1)
            da_target = torch.argmax(target[:, 0:2, :, :], dim=1)
            ll_target = torch.argmax(target[:, 2:4, :, :], dim=1)
            
            # Update metrics
            da_segment_results.addBatch(da_predict.cpu().numpy(), da_target.cpu().numpy())
            ll_segment_results.addBatch(ll_predict.cpu().numpy(), ll_target.cpu().numpy())
            
            # Clear cache periodically
            if i % 50 == 0:
                torch.cuda.empty_cache()
    
    return da_segment_results, ll_segment_results


def netParams(model):
    '''
    Calculate total parameters in model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
