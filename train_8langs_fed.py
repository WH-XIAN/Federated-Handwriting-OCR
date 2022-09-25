# -*- coding: utf-8 -*-
import argparse
from easydict import EasyDict as edict
import yaml
import os, sys
import copy
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split, ConcatDataset

import keys as alphabet
import lib.model.hw_mobile_net as hw_rec
import lib.model.crnn as crnn
import lib.config.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import hw_function as function
import lib.config.alphabets_new as alphabets_raw #  alphabet_8langs
from lib.config.utils.utils import model_info

from tensorboardX import SummaryWriter
import pdb
import Logger


def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")
    parser.add_argument('--cfg', help='experiment configuration filename', default='lib/config/hw_512_config.yaml', type=str)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.load(f)
        config = edict(config)
    # 给alphabet 赋值 用最新的 八种语言的 dict
    config.DATASET.ALPHABETS = alphabets_raw.alphabet_cn
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    print(config.MODEL.NUM_CLASSES)

    return config


def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0


def get_device(idx):
    n_gpu = torch.cuda.device_count()
    # device = torch.device("cuda:" + str(idx % n_gpu) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def sample_split(l, sl, num):
    n, r = divmod(l - sl, num)
    res = [n] * (num + 1)
    for i in range(r):
        res[i] += 1
    res[-1] = sl
    return res


def pretrain(config, train_loader, converter, model, criterion, device, logger):
    optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, config.TRAIN.BEGIN_EPOCH - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, config.TRAIN.BEGIN_EPOCH - 1
        )
    for startup_epoch in range(config.FED.STARTUP_EPOCH):
        msg = 'Pretrain on shared data, epoch {0}'.format(startup_epoch)
        logger.info(msg)
        for i, (inp, labels, idx) in enumerate(train_loader):
            inp = inp.to(device)
            preds = model(inp).cpu()
            batch_size = inp.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size) # timestep * batchsize
            loss = criterion(preds, text, preds_size, length)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()
        lr_scheduler.step()


def train(config, train_loader, converter, model, criterion, device, epoch, rank, logger, writer_dict=None, output_dict=None):

    batch_time = function.AverageMeter()
    data_time = function.AverageMeter()
    losses = function.AverageMeter()

    optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, config.TRAIN.BEGIN_EPOCH - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, config.TRAIN.BEGIN_EPOCH - 1
        )
    
    for local_epoch in range(config.FED.LOCAL_EPOCH):
        end = time.time()
        for i, (inp, labels, idx) in enumerate(train_loader):
            # measure data time
            data_time.update(time.time() - end)

            # labels = utils.get_batch_label(dataset, idx)
            inp = inp.to(device)
            
            # print('cat inp shape ', inp.shape)
        
            # print('inp shape {0} and its idx {1}'.format(inp[0].shape, idx))
            # inference
            preds = model(inp).cpu()

            # compute loss
            batch_size = inp.size(0)
            text, length = converter.encode(labels)                    # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
            # print('loss length', length)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size) # timestep * batchsize
            loss = criterion(preds, text, preds_size, length)
            
            # loss = topk(loss, k)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()
            losses.update(loss.item(), inp.size(0))
            batch_time.update(time.time()-end)
            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch/Rank/Local: [{0}_{1}_{2}][{3}/{4}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                        epoch, rank, local_epoch, i, len(train_loader), batch_time=batch_time,
                        speed=inp.size(0)/batch_time.val,
                        data_time=data_time, loss=losses)
                #print(msg)
                logger.info(msg)


                if writer_dict:
                    writer = writer_dict['writer']
                    global_steps = writer_dict['train_global_steps']
                    writer.add_scalar('train_loss', losses.avg, global_steps)
                    writer_dict['train_global_steps'] = global_steps + 1

            end = time.time()
            # if i >= 10: break
        lr_scheduler.step()

            
def main():
    # load config
    config = parse_arg()

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')
    logger = Logger.Logger(output_dict['log_file'])
    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # get device
    device = get_device(0)
    # create global and local models
    global_model = hw_rec.get_model(config)
    global_model = global_model.to(device)
    local_model = hw_rec.get_model(config)
    local_model = local_model.to(device)
    
    # define loss function
    criterion = torch.nn.CTCLoss(zero_infinity=True)

    last_epoch = config.TRAIN.BEGIN_EPOCH
    '''
    optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, config.TRAIN.BEGIN_EPOCH - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, config.TRAIN.BEGIN_EPOCH - 1
        )
    '''
    if config.TRAIN.FINETUNE.IS_FINETUNE:      
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        print('loading model: ', model_state_file)
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        # from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        global_model.cnn.load_state_dict(model_dict)
        if config.TRAIN.FINETUNE.FREEZE:
            for p in global_model.cnn.parameters():
                p.requires_grad = False

    elif config.TRAIN.RESUME.IS_RESUME:
        print('MOVE FAST and make things happen ====> lets resume to prev model training and deliver it')
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        
        if 'state_dict' in checkpoint.keys():
            print('loading pretrained weights: ', config.TRAIN.RESUME.FILE)
            try:
                global_model.load_state_dict(checkpoint['state_dict'])
            except:
                # print('remove last layer')
                preWeightDict = torch.load(model_state_file)['state_dict']##加入项目训练的权重
                modelWeightDict = global_model.state_dict()
                #pdb.set_trace()
                for k, v in preWeightDict.items():
                    name = k.replace('module.','') # remove `module.`
                    # if 'trans_layer' not in name: #不加载最后一层权重
                    #     modelWeightDict[name] = v
                    modelWeightDict[name] = v
                    # modelWeightDict[k] = v
                global_model.load_state_dict(modelWeightDict)
            last_epoch = checkpoint['epoch']
            if last_epoch > 0:
                last_epoch = 0
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            global_model.load_state_dict(checkpoint)

    global_model.register_backward_hook(backward_hook)
    local_model.register_backward_hook(backward_hook)
    model_info(global_model)
    
    train_dataset = get_dataset(config)(config, is_train=True)
    lt = len(train_dataset)
    share_lt = int(lt * config.FED.SHARE_RATE)
    train_dataset_split = random_split(train_dataset, sample_split(lt, share_lt, config.FED.NUM_USERS))
    train_loader = []
    for i in range(config.FED.NUM_USERS):
        train_loader_single = DataLoader(
                dataset=ConcatDataset([train_dataset_split[i], train_dataset_split[-1]]),
                batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
                shuffle=config.TRAIN.SHUFFLE,
                num_workers=config.WORKERS,
                pin_memory=config.PIN_MEMORY,
        )
        train_loader.append(train_loader_single)
    train_loader_share = DataLoader(
            dataset=train_dataset_split[-1],
            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY,
    )
    train_loader.append(train_loader_share)

    val_dataset = get_dataset(config)(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    best_acc = 0.01
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    num_active_users = int(np.ceil(config.FED.FRAC * config.FED.NUM_USERS))
    global_update = OrderedDict()

    user_bn_dict = []
    if config.MODEL.NORM == 'BatchNorm':
        pretrain(config, train_loader[-1], converter, global_model, criterion, device, logger)
        for i in range(config.FED.NUM_USERS):
            local_dict = OrderedDict()
            for k, v in global_model.state_dict().items():
                if 'running_mean' in k or 'running_var' in k:
                    local_dict[k] = v
            user_bn_dict.append(local_dict)

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):  
        user_idx = torch.arange(config.FED.NUM_USERS)[torch.randperm(config.FED.NUM_USERS)[:num_active_users]].tolist()
        pulled_model_dict = copy.deepcopy(global_model.state_dict())
        
        for k, v in global_model.state_dict().items():
            if 'weight' in k or 'bias' in k or config.MODEL.NORM == 'LayerNorm':
                global_update[k] = torch.zeros_like(v)
            else:
                global_update[k] = v
            
        for rank in range(num_active_users):
            local_loading_dict = copy.deepcopy(pulled_model_dict)
            if config.MODEL.NORM == 'BatchNorm':
                for k, v in user_bn_dict[user_idx[rank]].items():
                    local_loading_dict[k] = v
            local_model.load_state_dict(local_loading_dict)
            local_model.train()
            train(config, train_loader[user_idx[rank]], converter, local_model, criterion, device, epoch, rank, logger, writer_dict, output_dict) 
            for k, v in local_model.state_dict().items():
                if 'weight' in k or 'bias' in k or config.MODEL.NORM == 'LayerNorm':
                    global_update[k] += v / num_active_users
                elif 'running_mean' in k or 'running_var' in k:
                    user_bn_dict[user_idx[rank]][k] = v
                
        global_model.load_state_dict(global_update)
        global_model.eval()
        acc = function.validate(config, val_loader, val_dataset, converter, global_model, criterion, device, epoch, logger, writer_dict, output_dict)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best:", is_best)
        print("best acc is:", best_acc)
        # save checkpoint
        if is_best or (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "state_dict": global_model.state_dict(),
                    "epoch": epoch + 1,
                    "best_acc": best_acc,
                },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
            )
    
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
