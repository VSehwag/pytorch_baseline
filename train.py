from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import argparse
import importlib
import time
import logging
import json
from collections import OrderedDict
import importlib

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex (otherwise use DP accelerator.")
    
import models
import data
import trainers
import utils
from utils import *


def main():
    parser = argparse.ArgumentParser(description="Robust residual learning")
    
    parser.add_argument("--configs", type=str, default="./configs/configs_cifar.yml")
    parser.add_argument(
        "--results-dir", type=str, default="/data/data_vvikash/fall20/arjun_icml/trained_models/",
    )
    parser.add_argument("--exp-name", type=str, default="temp")
    parser.add_argument("--arch", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    # training
    parser.add_argument("--trainer", type=str, default="baseline", choices=("baseline", "adv"))
    parser.add_argument("--val-method", type=str, default="baseline", choices=("baseline", "adv"))
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--accelerator", type=str, default="dp", choices=("dp", "ddp"))
    parser.add_argument("--fp16", action="store_true", default=False, help="half precision training")

    # misc
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--ckpt", type=str, help="checkpoint path for pretrained classifier")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    
    args = update_args(parser.parse_args())
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    ngpus = torch.cuda.device_count() # Control available gpus by CUDA_VISIBLE_DEVICES only 
    print(f"Using {ngpus} gpus")
    args.distributed = (args.accelerator == "ddp") and ngpus > 1 # Need special care with ddp distributed training
    
    if args.fp16 and ngpus > 1:
        assert args.accelerator == "ddp", "half precision on multiple gpus supported only ddp mode"
    assert args.normalize == False, "Presumption for most code is that the pixel range is [0,1]"
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    #assert args.lr == 0.1 * (args.batch_size // 128), "Manully scale learning rate with 0.1*batch-size/128 rule"
    
    # seed cuda
    torch.manual_seed((args.local_rank+1)*args.seed)
    torch.cuda.manual_seed((args.local_rank+1)*args.seed)
    torch.cuda.manual_seed_all((args.local_rank+1)*args.seed)
    np.random.seed((args.local_rank+1)*args.seed)
    
    # create resutls dir (for logs, checkpoints, etc.)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    result_main_dir = os.path.join(args.results_dir, args.exp_name)
    result_sub_dir = os.path.join(result_main_dir, f"trial_{args.trial}")
    create_subdirs(result_sub_dir)
    
    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)
    
    # multi-gpu DDP
    if args.accelerator == "ddp":
        torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
        world_size = torch.distributed.get_world_size()
        print("world_size =", world_size)
        
        # Scale learning rate based on global batch size
        args.batch_size = args.batch_size // world_size
        args.workers = args.workers // world_size
        print(f"New per-gpu batch-size = {args.batch_size}, workers = {args.batch_size}")
    
    # create model + optimizer
    model = models.__dict__[args.arch](num_classes=args.num_classes).to(device).train()
    if args.ckpt is not None:
        d = fix_legacy_dict(torch.load(args.ckpt, map_location="cpu"))
        model.load_state_dict(d, strict=True)
        print(f"Mismatched keys {set(d.keys()) ^ set(model.state_dict().keys())}")
        print(f"model loaded from {args.ckpt}")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # half-precision support (Actually O1 in amp is mixed-precision)
    if args.fp16:
        #print("using apex synced BN")
        #model = apex.parallel.convert_syncbn_model(model)
        
        # O1 opt-level by default (O1 keeps batch_norm in float32)
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    
    # parallelization
    if ngpus > 1:
        print(f"Using multiple gpus")
        if args.accelerator == "dp":
            model = nn.DataParallel(model).to(device)
        elif args.accelerator == "ddp":
            model = DDP(model, delay_allreduce=True)
        else:
            raise ValueError("accelerator not supported")
    
    # dataloaders
    train_loader, train_sampler, val_loader, val_sampler, _, _, train_transform = data.__dict__[args.dataset](args.data_dir, batch_size=args.batch_size, mode=args.mode, normalize=args.normalize, size=args.size, workers=args.workers, distributed=args.distributed)
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader), 1e-4)
    criterion = nn.CrossEntropyLoss()
    best_prec = 0
    
    # Let's roll
    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        results_train = getattr(trainers, args.trainer)(model, device, train_loader, criterion, optimizer, lr_scheduler, epoch, args)
        results_val = getattr(utils, args.val_method)(model, device, val_loader, criterion, args, epoch)
        
        if args.local_rank == 0:
            # remember best prec@1 (only based on clean accuracy) and save checkpoint
            if args.val_method == "baseline":
                prec = results_val["top1"]
            elif args.val_method == "adv":
                prec = results_val["top1_adv"]
            else:
                raise ValueError()
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)

            d = {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec,
                "optimizer": optimizer.state_dict(),
            }

            save_checkpoint(
                d, is_best, result_dir=os.path.join(result_sub_dir, "checkpoint"),
            )
            
            logger.info(f"Epoch {epoch}, " + ", ".join(["{}: {:.3f}".format(k+"_train", v) for (k,v) in results_train.items()]+["{}: {:.3f}".format(k+"_val", v) for (k,v) in results_val.items()]))

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
