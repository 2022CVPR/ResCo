import argparse
import builtins
import os
import random
import time
import warnings
import pdb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

from models import resnet_imagenet_siam
from randaugment import rand_augment_transform
from datasets.imagenet_nori import ImageNet

from logger import _C as config
from logger import update_config, get_logger
from utils import AverageMeter, ProgressMeter, save_checkpoint, adjust_learning_rate, accuracy

import resco.loader
import resco.builder
import losses


def parse_args():
    parser = argparse.ArgumentParser(description='ResCo training')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()
    config.dist_url = "tcp://127.0.0.1:{}".format(random.randint(0, 20000) % 20000 + 6666)

    if config.debug:
        config.batch_size = 32
        config.workers = 0
        config.gpu = 0
        config.rank = 0
        config.world_size = 1
        config.seed = 0

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if config.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config.dist_url == "env://" and config.world_size == -1:
        config.world_size = int(os.environ["WORLD_SIZE"])

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    if config.debug:
        config.distributed = True
        config.multiprocessing_distributed = False

    ngpus_per_node = torch.cuda.device_count()
    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    config.gpu = gpu
    logger = get_logger(config, resume=False, is_rank0=(gpu==0))
    logger.info(str(config))

    # suppress printing if not master
    if config.multiprocessing_distributed and config.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if config.gpu is not None:
        logger.info("Use GPU: {} for training".format(config.gpu))

    if config.distributed:
        if config.dist_url == "env://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
    # create model


    logger.info("=> creating model '{}'".format(config.arch))
    model = resco.builder.ResCo(
        getattr(resnet_imagenet_siam, config.arch), pos_size_per_cls=config.pos_size_per_cls, 
        neg_size_per_cls=config.neg_size_per_cls, class_num=1000, dim=config.moco_dim)

    logger.info(model)
    
    if config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])

            if config.resume:
                state_dict = model.state_dict()
                state_dict_ssp = torch.load(config.resume, map_location=torch.device('cpu'))['state_dict']
                # logger.info(state_dict_ssp.keys())

                for key in state_dict.keys():
                    if key in state_dict_ssp.keys() and state_dict[key].shape == state_dict_ssp[key].shape:
                        state_dict[key]=state_dict_ssp[key]
                        logger.info(key + " *******loaded******* ")
                    else:
                        logger.info(key + " *******unloaded******* ")
                        
                model.load_state_dict(state_dict, strict=False)

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)


    cudnn.benchmark = True

    if config.dataset == 'imagenet':
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)

        augmentation_randncls = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0),
            transforms.RandomGrayscale(p=0.2),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(config.randaug_n, config.randaug_m), ra_params),
            transforms.ToTensor(),
            normalize,
        ]
        
        train_transform = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randncls)]
    
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
        train_dataset = ImageNet(root=config.data_dir, train=True, transform=train_transform)
        val_dataset = ImageNet(root=config.data_dir, train=False, transform=val_transform)
    
    logger.info(f'===> Training data length {len(train_dataset)}')
    logger.info(f'===> Evaluating data length {len(val_dataset)}')

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None),
        num_workers=config.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size * 4, shuffle=False,
        num_workers=config.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion_ce = nn.CrossEntropyLoss().cuda(config.gpu)
    criterion_joint = losses.SiamBalSfx_SiamBalQ(train_dataset.cls_num_list, \
                                         balsfx_n=config.balsfx_n, \
                                         queue_size_per_cls=config.queue_size_per_cls, \
                                         temperature=config.temperature, \
                                         con_weight=config.con_weight, \
                                         effective_num_beta=config.effective_num_beta).cuda(config.gpu)

    if config.evaluate:
        logger.info(" *******start evaluation******* ")
        validate(val_loader, model, criterion_ce, logger, config)
        return

    best_acc1 = 0
    acc1 = 0
    is_best = False
    
    for epoch in range(config.start_epoch, config.epochs + 1):
        if config.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(config, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion_joint, optimizer, epoch, logger, config)
        logger.info(" ")
        
        if epoch >= 0.97 * (config.epochs + 1):
            acc1 = validate(val_loader, model, criterion_ce, logger, config)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            logger.info(output_best)
                
        if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': config.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best, config.model_dir)



def train(train_loader, model, criterion, optimizer, epoch, logger, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_cls = AverageMeter('Cls_L', ':.3f')
    losses_cont = AverageMeter('Con_L', ':.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [losses_cls, losses_cont, losses, top1],
        logger,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images_org, images_con, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end) 

        if config.gpu is not None:
            images_org = images_org.cuda(config.gpu, non_blocking=True)
            target = target.cuda(config.gpu, non_blocking=True)
            images_con = images_con.cuda(config.gpu, non_blocking=True)

        # compute output
        sim_con, labels_con, logits_cls_q, logits_cls_k = model(im_q=images_org, im_k=images_con, labels_q=target, labels_k=target)
        loss_cls, loss_con, loss = criterion(sim_con, labels_con, logits_cls_q, logits_cls_k, target)

        acc1, _ = accuracy(logits_cls_q, target, topk=(1, 5))
        losses_cls.update(loss_cls.item(), logits_cls_q.size(0))
        losses_cont.update(loss_con.item(), logits_cls_q.size(0))
        losses.update(loss.item(), logits_cls_q.size(0))
        top1.update(acc1[0], logits_cls_q.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i)



def validate(val_loader, model, criterion, logger, config):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [losses, top1, top5],
        logger, 
        prefix='Eval: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if config.gpu is not None:
                images = images.cuda(config.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(config.gpu, non_blocking=True)

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

            if i % config.print_freq == 0:
                progress.display(i)

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg



if __name__ == '__main__':
    main()
