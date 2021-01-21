import os
import numpy as np
import math
import glob
from PIL import Image
from collections import OrderedDict
from easydict import EasyDict
import time
import shutil, errno
import yaml
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
import random
import pickle

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from autoattack import AutoAttack

from utils_adv import pgd_whitebox


def update_args(args):
    with open(args.configs) as f:
        new_args = EasyDict(yaml.load(f))
    
    for k, v in vars(args).items():
        if k in list(new_args.keys()):
            if v:
                new_args[k] = v
        else:
            new_args[k] = v
    
    return new_args


def display_vectors(images):
    if len(images) > 64:
        images = images[:64]
    if torch.is_tensor(images):
        images = np.transpose(images.cpu().numpy(), (0, 2, 3, 1))

    d = int(math.sqrt(len(images)))
    plt.figure(figsize=(8, 8))
    image = np.concatenate(
        [
            np.concatenate([images[d * i + j] for j in range(d)], axis=0)
            for i in range(d)
        ],
        axis=1,
    )
    if image.shape[-1] == 1:
        plt.imshow(image[:, :, 0], cmap="gray")
    else:
        plt.imshow(image)
        

def fix_legacy_dict(d):
    keys = list(d.keys())
    if "model" in keys:
        d = d["model"]
    if "state_dict" in keys:
        d = d["state_dict"]
    keys = list(d.keys())
    # remove multi-gpu module.
    if "module." in keys[1]:
        d = remove_module(d)
    return d
    
    
def save_checkpoint(state, is_best, result_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(result_dir, filename),
            os.path.join(result_dir, "model_best.pth.tar"),
        )


def create_subdirs(sub_dir):
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(os.path.join(sub_dir, "checkpoint"), exist_ok=True)


def write_to_file(file, data, option):
    with open(file, option) as f:
        f.write(data)


def clone_results_to_latest_subdir(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst)
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def write_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.val, global_step)

    def write_avg_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.avg, global_step)
            
            
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
def baseline(model, device, val_loader, criterion, args, epoch=0):
    """
        Evaluating on unmodified validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0 and args.local_rank == 0:
                progress.display(i)

        if args.local_rank == 0:
            progress.display(i)  # print final results

    result = {"top1": top1.avg, "top5":  top5.avg}
    return result


def adv(model, device, val_loader, criterion, args, epoch=0):
    """
        Evaluate on adversarial validation set inputs.
    """

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    adv_losses = AverageMeter("Adv_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    top1_adv = AverageMeter("Adv-Acc_1", ":6.2f")
    top5_adv = AverageMeter("Adv-Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, adv_losses, top1, top5, top1_adv, top5_adv],
        prefix="Test: ",
    )

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images
            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # adversarial images
            images = pgd_whitebox(
                model,
                images,
                target,
                device,
                args.epsilon,
                args.num_steps,
                args.step_size,
                0.,
                1.,
                is_random=True,
                distance=args.distance
            )

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            adv_losses.update(loss.item(), images.size(0))
            top1_adv.update(acc1[0], images.size(0))
            top5_adv.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0 and args.local_rank == 0:
                progress.display(i)

        if args.local_rank == 0:
            progress.display(i)  # print final results
    result = {"top1": top1.avg, "top5":  top5.avg, "top1_adv": top1_adv.avg, "top5_adv": top5_adv.avg}
    return result


def auto(model, device, val_loader, criterion, args, epoch=0):
    """
        Evaluate on atuo-attack adversarial validation set inputs.
    """

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    adv_losses = AverageMeter("Adv_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    top1_adv = AverageMeter("Adv-Acc_1", ":6.2f")
    top5_adv = AverageMeter("Adv-Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, adv_losses, top1, top5, top1_adv, top5_adv],
        prefix="Test: ",
    )

    # switch to evaluation mode
    model.eval()
    assert args.distance in ["linf", "l2"]
    
    adversary = AutoAttack(model, norm="Linf" if args.distance=="linf" else "L2", eps=args.epsilon)

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # clean images
            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            images = adversary.run_standard_evaluation(images, target, bs=len(images))
            
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            adv_losses.update(loss.item(), images.size(0))
            top1_adv.update(acc1[0], images.size(0))
            top5_adv.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0 and args.local_rank == 0:
                progress.display(i)

        if args.local_rank == 0:
            progress.display(i)  # print final results

    result = {"top1": top1.avg, "top5":  top5.avg, "top1_adv": top1_adv.avg, "top5_adv": top5_adv.avg}
    return result