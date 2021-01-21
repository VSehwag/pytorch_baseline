import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from utils import AverageMeter, ProgressMeter
from utils import accuracy
from utils_adv import pgd_whitebox
from apex import amp
from utils_adv import trades_loss
    
def baseline(model, device, dataloader, criterion, optimizer, lr_scheduler=None, epoch=0, args=None):
    if args.local_rank == 0:
        print(" ->->->->->->->->->-> One epoch with Baseline natural training <-<-<-<-<-<-<-<-<-<-")
    
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    
    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)
            
        # basic properties of training
        if i == 0 and args.local_rank == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )

        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank == 0:
            progress.display(i)
    result = {"top1": top1.avg, "top5":  top5.avg}
    return result


def adv(model, device, dataloader, criterion, optimizer, lr_scheduler=None, epoch=0, args=None):
    if args.local_rank == 0:
        print(" ->->->->->->->->->-> One epoch with Adversarial (Trades) training <-<-<-<-<-<-<-<-<-<-")
        
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    top1_adv = AverageMeter("Acc_1_adv", ":6.2f")
    top5_adv = AverageMeter("Acc_5_adv", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top1, top5, top1_adv, top5_adv],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()
    
    for i, data in enumerate(dataloader):
        images, target = data[0].to(device), data[1].to(device)

        # basic properties of training
        if i == 0 and args.local_rank == 0:
            print(
                images.shape,
                target.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(images).data.cpu().numpy(),
                    torch.max(images).data.cpu().numpy(),
                )
            )

        # calculate robust loss
        loss, logits, logits_adv = trades_loss(
            model=model,
            x_natural=images,
            y=target,
            device=device,
            optimizer=optimizer,
            step_size=args.step_size,
            epsilon=args.epsilon,
            perturb_steps=args.num_steps,
            beta=args.beta,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            distance=args.distance,
        )

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        acc1_adv, acc5_adv = accuracy(logits_adv, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1_adv.update(acc1_adv[0], images.size(0))
        top5_adv.update(acc5_adv[0], images.size(0))

        optimizer.zero_grad()
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.local_rank == 0:
            progress.display(i)
    result = {"top1": top1.avg, "top5":  top5.avg, "top1_adv": top1_adv.avg, "top5_adv": top5_adv.avg}
    return result