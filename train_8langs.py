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

def main():

    # load config
    config = parse_arg()

    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')
    logger = Logger.Logger(output_dict['log_file'])

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    model = hw_rec.get_model(config)
    
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss(zero_infinity=True)

    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    if config.TRAIN.FINETUNE.IS_FINETUNE:
        
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        print('loading model: ', model_state_file)
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        model.cnn.load_state_dict(model_dict)
        if config.TRAIN.FINETUNE.FREEZE:
            for p in model.cnn.parameters():
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
                model.load_state_dict(checkpoint['state_dict'])
            except:
                # print('remove last layer')
                preWeightDict = torch.load(model_state_file)['state_dict']##加入项目训练的权重
                modelWeightDict = model.state_dict()
                #pdb.set_trace()
                for k, v in preWeightDict.items():
                    name = k.replace('module.','') # remove `module.`
                    # if 'trans_layer' not in name: #不加载最后一层权重
                    #     modelWeightDict[name] = v
                    modelWeightDict[name] = v
                model.load_state_dict(modelWeightDict)
            last_epoch = checkpoint['epoch']
            if last_epoch > 0:
                last_epoch = 0
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)

    model.register_backward_hook(backward_hook)
    model_info(model)
    train_dataset = get_dataset(config)(config, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

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
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch, logger, writer_dict, output_dict)
        lr_scheduler.step()

        acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch, logger, writer_dict, output_dict)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best:", is_best)
        print("best acc is:", best_acc)
        # save checkpoint
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "best_acc": best_acc,
            },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )

    writer_dict['writer'].close()

if __name__ == '__main__':
    main()
