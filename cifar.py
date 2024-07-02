'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017 ( •̀ ω •́ )y
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import pdb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
# import models.cornet as cornet
from copy import deepcopy
# import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from utils import  Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tqdm import tqdm
from my_utils import *
import wandb
from datetime import datetime
import utils.myModels as myModels
import utils.myModels2 as myModels2
import utils.myModels3 as myModels3
import utils.myModels4 as myModels4
from tent.robustbench.model_zoo.architectures.wide_resnet import WideResNet



model_names = sorted([name for name in models.__dict__
    if name.islower() and not name.startswith("__")
     and  callable(models.__dict__[name])]
                     )


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=4, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--opt', choices = ['SGD','AdamW'], default = 'SGD')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
# parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    # help='Decrease learning rate at these epochs.')
parser.add_argument('--lr-scheduler', default='StepLR', choices = ['StepLR','CosALR'])
parser.add_argument('--step_size', default = 60, type = int)
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56_SDN',  #resnet56_SDNb3b5OutputLong
#                     choices=model_names,
#                     help='model architecture: ' +
#                     ' | '.join(model_names) +
#                     ' (default: resnet18)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56_SDN',  #resnet56_SDNb3b5OutputLong
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8,
                    help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4,
                    help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12,
                    help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2,
                    help='Compression Rate (theta) for DenseNet.')
parser.add_argument('--add_ic', type=str, default='b2b3b4', help='Model depth.')

parser.add_argument('--interOuts', type=int, default=0, help='interOuts')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed',default=14)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu-id', default= '0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many epochs to wait before logging training status')
parser.add_argument('--times',type = int, nargs = '+')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
# os.environ["WANDB_API_KEY"] = 'KEY'
# os.environ["WANDB_MODE"] = "offline"
# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",
#
#     # track hyperparameters and run metadata
#     config={
#         "learning_rate": args.lr,
#         "architecture": args.arch,
#         "dataset": "CIFAR-10",
#         "epochs": 200,
#     }
# )


def main():
    global best_acc
    global best_epoch
    best_epoch = 0
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    
    args.checkpoint = f'{args.checkpoint}/{args.dataset}/{args.arch}_w{args.widen_factor}'
    if args.arch.startswith('cornet'):
        str_append = 'times'
        for i in args.times:
            str_append += f'_{i}'
        args.checkpoint += '/'+str_append
        
    args.checkpoint += f'/{args.opt}/bs{args.train_batch}_lr{args.lr}_wd{args.weight_decay}'

    current_time = datetime.now().time()
    print("Current Time:", current_time)
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")

    if args.lr_scheduler == 'CosALR':
        args.checkpoint += f'CosALR_{args.epochs}'
    elif args.lr_scheduler == 'StepLR':
        args.checkpoint += f'StepLR_{args.step_size}_{args.gamma}'
    else:
        raise NotImplementedError(args.lr_scheduler)

    # finetune
    args.checkpoint += f'/seed_{args.manualSeed}/finetuneResout{args.interOuts}/addIc_{args.add_ic}ResOut'

    # args.checkpoint += f'/NOOutputs/seed_{args.manualSeed}'

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    args.save_name = args.checkpoint.replace('/','-')
    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./data', train=True,
                          download=True, transform=transform_train)
    trainloader = data.DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False,
                         download=False, transform=transform_test)
    testloader = data.DataLoader(
        testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    # pdb.set_trace()
    print("==> creating model '{}'".format(args.arch))
    # if args.arch.startswith('resnext'):
    #     model = models.__dict__[args.arch](
    #         cardinality=args.cardinality,
    #         num_classes=num_classes,
    #         depth=args.depth,
    #         widen_factor=args.widen_factor,
    #         dropRate=args.drop,
    #     )
    # elif args.arch.startswith('densenet'):
    #     model = models.__dict__[args.arch](
    #         num_classes=num_classes,
    #         depth=args.depth,
    #         growthRate=args.growthRate,
    #         compressionRate=args.compressionRate,
    #         dropRate=args.drop,
    #     )
    # elif args.arch.startswith('wrn'):
    #     model = models.__dict__[args.arch](
    #         num_classes=num_classes,
    #         depth=args.depth,
    #         widen_factor=args.widen_factor,
    #         dropRRate=args.drop,
    #     )
    # elif args.arch.startswith('resnet'):
    #     model = models.__dict__[args.arch](
    #         num_classes=num_classes,
    #         depth=args.depth,
    #         block_name=args.block_name,
    #     )
    # else:
    #     model = models.__dict__[args.arch](num_classes=num_classes)

    # model = myModels.resnet56_with_dropout(dropout_prob=args.drop, num_classes=10)
    # model = myModels.resnet32NewEarly_with_dropout(num_classes=10, dropout_prob=args.drop)

    if args.add_ic == "b3b4b5b6":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0]
        ]
    elif args.add_ic == "b4b5b6":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0]
        ]
    elif args.add_ic == "b5b6":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0]
        ]
    elif args.add_ic == "b3b5":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0]
        ]
    elif args.add_ic == "b2b3b4b5b6":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0]
        ]
    elif args.add_ic == "b1b2b3b4b5b6":
        add_ic = [
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0]
        ]
    elif args.add_ic == "b2b3b4":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    elif args.add_ic == "b2b4b6":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0]
        ]
    model = myModels4.resnetSDN(add_ic, args.interOuts)

    # add_ic = [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # ]
    # model = myModels3.resnetSDN(add_ic)


    writer = SummaryWriter(log_dir=f'tensorLog/{args.arch}FinetuneResOut/add_ic_{args.add_ic}/cifar10_lr{args.lr}_seed{args.manualSeed}_interOuts{args.interOuts}')

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel()
          for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss() 
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.opt)  
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise NotImplementedError(args.lr_scheduler)
    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        best_epoch = checkpoint['best_epoch']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'),
                        title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss',
                         'Valid Loss', 'Train Acc.', 'Valid Acc.','Best Valid Acc.','Best Valid Epoch'])
        with open(args.checkpoint + '/net.txt','w') as f:
            f.write(str(model))
        with open(os.path.join(args.checkpoint, 'args.txt'), 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))
        # logger_ly = Logger(os.path.join(args.checkpoint, 'log_ly.txt'), title=title)
        # logger_ly.set_names(['loss0','loss1','loss2','loss3','loss4'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(
            testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    last_state=deepcopy(model.state_dict())

    # finetune
    for module in model.modules():
        if isinstance(module, myModels4.BasicBlockWOutput) and module.output is not None:
            # interoutput1
            # for param in module.output.linear.parameters():
            # interoutput2
            for param in module.output.parameters():
                param.requires_grad = False
    
    for epoch in range(start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' %
              (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(
            trainloader, model, criterion, optimizer, epoch, use_cuda, writer)
        writer.add_scalar('train_acc', train_acc, epoch)
        # writer.add_scalar(tag="train_acc",  # 可以暂时理解为图像的名字
        #                   scalar_value=top1.avg,  # 纵坐标的值
        #                   global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
        #                   )
        writer.add_scalar(tag="train_loss",  # 可以暂时理解为图像的名字
                          scalar_value=train_loss,  # 纵坐标的值
                          global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                          )
        lr_scheduler.step()

        test_loss, test_acc = test(
            testloader, model, criterion, epoch, use_cuda, writer)

        writer.add_scalar(tag="test_acc",  # 可以暂时理解为图像的名字
                          scalar_value=test_acc,  # 纵坐标的值
                          global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                          )
        writer.add_scalar(tag="test_loss",  # 可以暂时理解为图像的名字
                          scalar_value=test_loss,  # 纵坐标的值
                          global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                          )

        optimizer.step()
        # losses_layers=cal_layer_loss(last_state,model.state_dict())
        last_state=deepcopy(model.state_dict())
        # append logger file
        logger.append([optimizer.param_groups[0]['lr'], train_loss,
                      test_loss, train_acc, test_acc,best_acc,best_epoch])
        # logger_ly.append([losses_layers[0],losses_layers[1],losses_layers[2],losses_layers[3],losses_layers[4]])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best:
            best_epoch = epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler':lr_scheduler.state_dict(),

        }, is_best, checkpoint=args.checkpoint)

        # if (epoch % args.log_interval == 0):
        #     save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'acc': test_acc,
        #     'best_acc': best_acc,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best, checkpoint=args.checkpoint,filename=f'epoch{epoch}.pth.tar')
        #     ind_ = 0
        #     for fm in model.module.featuremaps:   
        #         for j in range(5):
        #             mkdir_p(args.checkpoint+f'/featuremaps/{j}')
        #             fn = args.checkpoint+f'/featuremaps/{j}/epoch{epoch}_ly{ind_}.png'
        #             out = torchvision.utils.make_grid(fm[j])
        #             feature_imshow(out, fn)
        #         ind_ += 1
        # Calculate layer difference loss
    logger.close()
    # logger_ly.close()
    # logger.plot()
    # savefig(os.path.join(args.checkpoint, 'log.eps'))
    # logger_ly.plot()
    # savefig(os.path.join(args.checkpoint, 'log_ly.eps'))
    # plot_module_loss(args)
        
    print('Best acc:')
    print(best_acc)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda, writer=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    output1Prec1, output2Prec1 = 0, 0
    prec1List = []
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(
            inputs), torch.autograd.Variable(targets)

        outputList = model(inputs)
        # compute output earlyOutPut
        # outputs, output1, output2 = model(inputs)
        # allOutputs = output1*args.A + output2*args.B + outputs*args.C
        # loss = criterion(outputs, targets)
        loss = 0
        max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1])
        # max_coeffs = np.array([0, 0, 1])
        cur_coeffs = 0.01 + epoch * (max_coeffs / args.epochs)  # to calculate the tau at the currect epoch
        cur_coeffs = np.minimum(max_coeffs, cur_coeffs)

        for ic_id in range(model.module.num_output):
            cur_output = outputList[ic_id]
            if ic_id < model.module.num_output - 1:
                cur_loss = 0 * criterion(cur_output, targets)
            else:
                cur_loss = 1 * criterion(cur_output, targets)
            # cur_loss = float(cur_coeffs[ic_id]) * criterion(cur_output, targets)

            loss += cur_loss
        if batch_idx == len(trainloader) - 1:
            for idx in range(model.module.num_output):
                prec1List.append(accuracy(outputList[idx].data, targets.data, topk=(1, 5)))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputList[-1].data, targets.data, topk=(1, 5))

        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        print(bar.suffix)
        # log metrics to wandb
        bar.next()
    bar.finish()
    # wandb.log({"train_acc": top1.avg, "train_loss": losses.avg})
    # for idx in range(len(prec1List)):
    #     writer.add_scalar(f'train_Output{idx}acc', prec1List[idx][0], epoch)

    # writer.add_scalar('train_Output1acc', output1Prec1/len(inputs), epoch)
    # writer.add_scalar('train_Output2acc', output2Prec1/len(inputs), epoch)
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda, writer=None):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    top1List = []
    for idx in range(model.module.num_output):
        top1List.append(AverageMeter())
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(
            inputs, volatile=True), torch.autograd.Variable(targets)

        outputsList = model(inputs)
        # compute output
        # outputs, output1, output2 = model(inputs)
        # allOutputs = output1*args.A + output2*args.B + outputs*args.C
        loss = criterion(outputsList[-1], targets)

        for output_id in range(model.module.num_output):
            cur_output = outputsList[output_id]
            prec1, prec5 = accuracy(cur_output.data, targets, topk=(1, 5))
            top1List[output_id].update(prec1, inputs.size(0))


        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputsList[-1].data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        # log metrics to wandb
        print(bar.suffix)
        bar.next()
    bar.finish()

    top1_accs = []
    for output_id in range(model.module.num_output):
        top1_accs.append(top1List[output_id].avg.data.cpu().numpy()[()])
        writer.add_scalar(f'test_Output{output_id+1}Acc', top1_accs[output_id], epoch)
    print("top1_accs", top1_accs)
    # wandb.log({"test_acc": top1.avg, "test_loss": losses.avg})
    # writer.add_scalar('test_Output1acc', output1Prec1/len(inputs), epoch)
    # writer.add_scalar('test_Output2acc', output2Prec1/len(inputs), epoch)
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint_.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, 'model_best.pth.tar'))

def save_checkpoint0(state, is_best, checkpoint='checkpoint', filename='checkpoint_.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, 'model_best.pth.tar'))

# def adjust_learning_rate(optimizer, epoch):
#     global state
#     if epoch in args.schedule:
#         state['lr'] *= args.gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
