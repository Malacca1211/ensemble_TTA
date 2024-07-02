from __future__ import print_function
import logging
import pdb
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

from tent.robustbench.data0 import load_cifar10c
from tent.robustbench.model_zoo.enums import ThreatModel
from tent.robustbench import load_model

import tent.tent as tent
from tent.tent import softmax_entropy
import tent.norm as norm
import argparse
import time
import os
import models.cifar as models
from tent.conf import cfg, load_cfg_fom_args
from utils.myModels import ResNetWithDropout, ResNet56, ResNet56EarlyOutput
import utils.myModels2 as myModels2
import utils.myModels3 as myModels3
import utils.myModels4 as myModels4
import torch.nn as nn
import my_utils
import math

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction


weights = []
err_means = []

# 打开文件并逐行读取数据
with open('RankTTA_norm_accuracies.txt', 'r') as file:
    for line in file:
        # 移除行末尾的换行符并按空格分割字符串
        line = line.strip().split()
        # 获取weight部分并转换为整数列表，移除逗号
        weight = [int(x.strip(',')) for x in line[1:8]]
        # 将weight列表添加到weights中
        weights.append(weight)
        # 获取err_mean部分并转换为浮点数
        err_mean = float(line[-1][:-1])  # 去除末尾的百分号并转换为浮点数
        # 将err_mean添加到err_means中
        err_means.append(err_mean)

    # 打印结果以检查
print("weights =", weights)
print("err_means =", err_means)


# 定义目标函数，接受权重值列表作为输入
def guassTarget(**weights_dict):
    weights_list = [weights_dict['x{}'.format(i)] for i in range(len(weights_dict))]
    return -10

# 创建 BayesianOptimization 对象
optimizer = BayesianOptimization(guassTarget, {'x{}'.format(i): (0, 1) for i in range(len(weights[0]))}, random_state=20)

# 创建 UtilityFunction 的实例
utility = UtilityFunction(kind="ucb", kappa=5, xi=0.0)

# 将实用程序函数传递给 maximize 方法
optimizer.maximize(init_points=10, n_iter=0, acquisition_function=utility)



torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

model_names = sorted([name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])] )

def parse_weights(weights_str):
    try:
        weights = [float(x) for x in weights_str.strip('[]').split(',')]
        return weights
    except ValueError:
        raise argparse.ArgumentTypeError("Weights must be comma-separated floats in square brackets.")


parser = argparse.ArgumentParser(description='tent')
parser.add_argument('--model-paths', nargs='+', default=[
      'finetune/cifar10/resnet56/finetuneCheckpoint/b1b2b3b4b5b6ResOut2/train30epochs_lr_0.01ModelTrain/outputIndex_6_bestModel'
])
parser.add_argument('--arch', '-a', metavar='ARCH',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)',default='resnet')
parser.add_argument('--depth', type=int, default=56, help='Model depth.')
parser.add_argument('--reset', type=bool, default=True, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cfg',dest='cfg_file', default='tent/cfgs/tent.yaml', type=str)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                      help="See conf.py for all options")
parser.add_argument('--out-dir', default='results_resemble/cifar10', type=str)
parser.add_argument('--ensemble-mode', default='mean_softmax', type=str, choices=['mean_softmax','majority_vote'])
parser.add_argument('--eval-source', default = False, action='store_true')
parser.add_argument('--dp', type=float, default=0, help='')
parser.add_argument('--InferT', type=int, default=10, help='')
parser.add_argument('--thredhold', type=float, default=2, help='')
parser.add_argument('--UseInfer', type=float, default=2, help='')
parser.add_argument('--tentLr', type=float, default=5.12e-05, help='')
parser.add_argument("--weights", type=parse_weights, help="Array of weights [w1, w2, ..., wn]")
parser.add_argument('--interOuts', type=int, default=2, help='')

args = parser.parse_args()


# 定义解析函数，将输入字符串解析为数组


class TwoLayerFullyConnectedNN(nn.Module):
    def __init__(self):
        super(TwoLayerFullyConnectedNN, self).__init__()
        self.weight = nn.Parameter(torch.rand(7, requires_grad=True))  # 定义可学习参数，并初始化在0到1之间

    def forward(self, x):
        # 将输入向量的每个元素分别与权重相乘

        weighted_sum = torch.zeros_like(x[0])
        for i, tensor in enumerate(x):
            weighted_sum += self.weight[i] * tensor
        #
        output = weighted_sum
        return output
def accuracy(model: torch.nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 100,
                   device: torch.device = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)
    mis_classified = []
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr)
            acc += (output.max(1)[1] == y_curr).float().sum()
            mis_classified.append((output.max(1)[1] != y_curr).nonzero()+counter * batch_size)

    return acc.item() / x.shape[0], mis_classified


def setup_single_model(base_model,optimizer=None):
    print('==> Loading checkpoint..')
    base_model=base_model.cuda()    # configure model
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "norm":
        logger.info("test-time adaptation: NORM")
        model = setup_norm(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "tent_mean_out":
        logger.info("test-time adaptation: TENT_MEAN_OUT")
        model = setup_tent_mean_output(base_model,optimizer=optimizer)

    return model


def evaluate(args,description):
    models = []
    # if cfg.MODEL.ADAPTATION != "tent_mean_out":
    for model_path in args.model_paths:
        if 'resnet110' in model_path:
            depth=110
        elif 'resnet56' in model_path:
            depth=56
            # dropout
            # models.append(setup_single_model(load_singleRes56Dropout_model(model_path, depth, num_classes)))
            # continue
        elif 'resnet34' in model_path:
            depth=34
            # models.append(setup_single_model(load_singleDropout_model(model_path, depth, num_classes)))
            continue
        elif 'resnet32' in model_path:
            depth=32
            # models.append(setup_single_model(load_singleRes32EarlyOutputDropout_model(model_path, depth, num_classes)))
            # models.append(setup_single_model(load_singleRes56Dropout_model(model_path, depth, num_classes)))
            continue
        else:
            raise Exception(f'No such depth! \n {model_path}')
        models.append(setup_single_model(load_singleRes56SDNmodel(model_path)))

    all_weights = my_utils.generate_all_weights(nums_out)

    # all_weights = [[0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 1, 1], [1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1]]

    all_weights = [[0.1862, 0.0899, 0.0909, 0.0621, 0.0962, 0.0656, 0.2182]]
    all_weights = [[1, 1, 1, 0, 1, 1, 1]]
    print(all_weights, f"{cfg.MODEL.ADAPTATION}", f"{args.tentLr}" )

    # 创建神经网络实例
    modelW = TwoLayerFullyConnectedNN()

    # 定义损失函数和优化器
    criterionW = nn.CrossEntropyLoss()  # 使用均方误差损失函数
    optimizerW = optim.SGD(modelW.parameters(), lr=0.01)  # 使用随机梯度下降优化器


    # 找到最后一个斜杠的索引
    last_slash_index = model_path.rfind('/')

    # 使用切片删除最后一个斜杠及其后面的部分
    checkPath = model_path[:last_slash_index+1]

    # checkPath = f'finetune/cifar10/resnet56/finetuneCheckpoint/b1b2b3b4b5b6ResOut{args.interOuts}/train50epochs_lr_0.01_step40_gamma0.1/'
    tentFilename = 'TTA_tent_accuracies.txt'
    normFilename = 'TTA_norm_accuracies.txt'
    sourceFilename = 'TTA_source_accuracies.txt'
    # fileName = logOutputAcc + f"/{normFilename}"
    fileName = checkPath + f"{normFilename}"
    with open(f"{normFilename}", "w") as file:
        file.write("TTA_accuracies:\n")
    for _ in range(6):
    # evaluate on each severity and type of corruption in turn
        next_point = optimizer.suggest(utility)
    for severity in cfg.CORRUPTION.SEVERITY:
            err_sum=0
            lossW = 0
            for corruption_type in cfg.CORRUPTION.TYPE:
                x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                               severity, cfg.DATA_DIR, False,
                                               [corruption_type])
                x_test, y_test = x_test.cuda(), y_test.cuda()
                x = x_test
                y = y_test
                batch_size = cfg.TEST.BATCH_SIZE
                device = x.device
                acc = 0.
                acc_this = 0.
                n_batches = math.ceil(x.shape[0] / batch_size)
                mis_classified = []
                mis_classified_this = []
                start_time = time.time()
                T = args.InferT

                outputAll = []

                with torch.enable_grad():
                # with torch.no_grad():
                    for counter in range(n_batches):
                        x_curr = x[counter * batch_size:(counter + 1) *
                                batch_size].to(device)
                        y_curr = y[counter * batch_size:(counter + 1) *
                                batch_size].to(device)
                        output = 0.

                        for model in models:
                            output = model(x_curr)

                        weights = list(next_point.values())
                        # 计算加权平均
                        weighted_sum = torch.zeros_like(output[0])
                        total_weight = sum(weights)
                        for i, tensor in enumerate(output):
                            weighted_sum += weights[i] * tensor
                        #
                        output = weighted_sum
                        output = weighted_sum / total_weight
                        output = output / len(models)

                        # outputW = modelW(output)
                        # lossW = criterionW(outputW, y_curr)
                        # # 反向传播和权重更新
                        # optimizerW.zero_grad()  # 梯度清零
                        # lossW.backward()  # 反向传播
                        # optimizerW.step()  # 权重更新
                        # # 将权重限制在0到1之间
                        # modelW.weight.data.clamp_(0, 1)
                        # print(f"================{modelW.weight.data}")
                        # output = outputW

                        acc += (output.max(1)[1] == y_curr).float().sum()

                        mis_classified.append(((output.max(1)[1] != y_curr).nonzero() + counter * batch_size).squeeze().tolist())
                # my_utils.calculate_confidence(outputAll, y_test, corruption_type)
                acc = acc.item() / x.shape[0]
                err = 1. - acc
                err_sum += err
                logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")

            target = guassTarget(**next_point)
            optimizer.register(params=next_point, target=target)
            print(target, next_point)

            err_mean = err_sum/len(cfg.CORRUPTION.TYPE)
            logger.info(f"error % [mean{severity}]: {err_mean:.2%}")
            # my_utils.save_EarlyoutModel_and_error(f'{err_mean:.2%}', len(models), args.model_paths[0], args.dp)
            with open(fileName, "a") as file:
                # tent
                # file.write("Weight %s, Lr %s,err_mean: %.2f%%\n" % (', '.join(map(str, weights)),  f"{args.tentLr}", err_mean * 100))
                # norm
                file.write("Weight %s, err_mean: %.2f%%\n" %(', '.join(map(str, modelW.weight.data)), err_mean * 100))
                # source
                # file.write("nums_models %s, err_mean: %.2f%%\n" % (nums_model, err_mean * 100))



def load_source_data(args):
    # print('==> Preparing dataset %s' % args.dataset)
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
    # if args.dataset == 'cifar10':
    dataloader = datasets.CIFAR10
    num_classes = 10
    # else:
    #     dataloader = datasets.CIFAR100
    #     num_classes = 100

    trainset = dataloader(root='./data', train=True,
                          download=True, transform=transform_train)
    trainloader = data.DataLoader(
        trainset, batch_size= cfg.TEST.BATCH_SIZE, shuffle=True )

    testset = dataloader(root='./data', train=False,
                         download=False, transform=transform_test)
    testloader = data.DataLoader(
        testset, batch_size= cfg.TEST.BATCH_SIZE, shuffle=False )
    return trainloader, testloader


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_norm(model):
    """Set up test-time normalization adaptation.

    Adapt by normalizing features with test batch statistics.
    The statistics are measured independently for each batch;
    no running average or other cross-batch estimation is used.
    """
    norm_model = norm.Norm(model)
    norm_model = tent.configure_model(norm_model)
    logger.info(f"model for adaptation: %s", model)
    stats, stat_names = norm.collect_stats(model)
    logger.info(f"stats for adaptation: %s", stat_names)
    return norm_model


def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)

    optimizer = setup_optimizer(params,args.tentLr)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model

def setup_tent_mean_output(model, optimizer = None):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    if optimizer is None:
        optimizer = setup_optimizer(params)
    else:
        optimizer.add_param_group({'params': params})
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model,optimizer

def setup_tent_all(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(True)
    params=model.parameters()
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: All")
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model

def setup_optimizer(params, tentLr):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=tentLr,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=tentLr,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError

import time
import os
import sys

def load_single_model(model_path,depth, num_classes, image_size):
    if not os.path.exists(model_path):
            # os.makedirs(record_path)
        raise Exception(f'No such directory! \n {model_path}')

    if args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=depth,
            block_name=args.block_name,
        )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)


    print(model)

    print('==> Resuming from checkpoint..')
    ckpt_path= model_path +'/model_best.pth.tar'
    try:
        checkpoint = torch.load(ckpt_path)
    except:
        # try:
        #     ckpt_path= model_path+'/checkpoint.pth.tar'
        #     checkpoint = torch.load(ckpt_path)
        # except:
            raise Exception(f'No such file! \n {ckpt_path}')



    start_epoch = checkpoint['epoch']
    model=torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    return model



def load_singleRes56SDNOriginalmodel(model_path):
    add_ic = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    model = myModels3.resnetSDN(add_ic=add_ic)
    print('==> Resuming from checkpoint..')
    ckpt_path = model_path + '/model_best.pth.tar'
    try:
        checkpoint = torch.load(ckpt_path)
    except:
        # try:
        #     ckpt_path= model_path+'/checkpoint.pth.tar'
        #     checkpoint = torch.load(ckpt_path)
        # except:
        raise Exception(f'No such file! \n {ckpt_path}')
    start_epoch = checkpoint['epoch']
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_singleRes56SDNmodel(model_path):
    global nums_out, logOutputAcc
    stradd_ic = model_path.split('finetuneCheckpoint/')[1].split('/')[0]
    nums_out = stradd_ic.count("b") + 1

    # stradd_ic, nums_out = "b1b2b3b4b5b6", 7

    logOutputAcc = '/'.join(model_path.split('/')[:-1])

    if "NewOut" in stradd_ic:
        stradd_ic = stradd_ic[:-6]
    if "ResOut" in stradd_ic and args.interOuts < 10:
        stradd_ic = stradd_ic[:-7]
    elif "ResOut" in stradd_ic and args.interOuts > 9:
        stradd_ic = stradd_ic[:-8]
    if stradd_ic == "b3b4b5b6":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0]
        ]
    elif stradd_ic == "b4b5b6":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0]
        ]
    elif stradd_ic == "b5b6":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0]
        ]
    elif stradd_ic == "b3b5":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0]
        ]
    elif stradd_ic == "b2b3b4b5b6":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0]
        ]
    elif stradd_ic == "b1b2b3b4b5b6":
        add_ic = [
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0]
        ]
    elif stradd_ic == "b2b4b6":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0]
        ]
    elif stradd_ic == "b2b3b4":
        add_ic = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    # 两个MODEl现在 model2自定义output、3原始output
    # model = myModels2.resnetSDN(add_ic=add_ic)
    model = myModels4.resnetSDN(add_ic=add_ic, interOuts=args.interOuts)
    print(model)

    print('==> Resuming from checkpoint..')
    ckpt_path = model_path + '/model_best.pth.tar'
    try:
        checkpoint = torch.load(model_path)
    except:
        # try:
        #     ckpt_path= model_path+'/checkpoint.pth.tar'
        #     checkpoint = torch.load(ckpt_path)
        # except:
        raise Exception(f'No such file! \n {ckpt_path}')
    start_epoch = checkpoint['epoch']
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def main():
    # random time string
    suffix_ = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    torch.autograd.set_detect_anomaly(True)
    # tta_method=args.cfg_file.split('/')[-1].split('.')[0]
    args.save_path = args.out_dir
    # os.stdout = Logger(f'{args.save_path}/'+cfg.LOG_DEST.replace('.txt','.log'))
    description = '"CIFAR-10-C evaluation.'
    load_cfg_fom_args(args,description)
    print(f'optim method is {cfg.OPTIM.METHOD}')
    print(f'optim lr is {cfg.OPTIM.LR}')
    # gpu_tracker.track()

    global best_prec1
    best_prec1 = 0
    global val_acc
    val_acc = []

    global image_size
    global num_classes
    image_size = 32 #for cifar10
    num_classes= 10
    base_models = []
    # for model_path in args.model_paths:
    #     base_models.append(load_single_model(model_path,num_classes, image_size))

    evaluate( args,'"CIFAR-10-C evaluation.')
    # gpu_tracker.track()

if __name__ == '__main__':

    main()
