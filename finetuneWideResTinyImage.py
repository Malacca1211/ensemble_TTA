import copy

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import utils.myModels4 as myModels4
import torch.nn as nn
import os
import errno
import argparse
from torch.optim import lr_scheduler
# from tent.robustbench.model_zoo.architectures.wide_resnet_SDN import WideResNet,NetworkBlock
from tent.robustbench.model_zoo.architectures.wide_resnet_SDNDuotou import resnetSDN, NetworkBlock
from PIL import Image
from tinyImageDataProcess import TinyImageNet

parser = argparse.ArgumentParser(description='PyTorch tiny Imagenet Training')
# Datasets
parser.add_argument('-m', '--model_path', default='checkpoint/tinyImagenet/wideResnet_SDN_w1/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_20/finetuneResout0/Wide1/addIc_b1-b3ResOut', type=str)
parser.add_argument('--interOuts', type=int, default=0, help='interOuts')
parser.add_argument('--epochs', type=int, default=20, help='epochs')
parser.add_argument('--step', type=int, default=40, help='step')
parser.add_argument('--seed', type=int, default=20, help='seed')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
parser.add_argument('--lr', type=float, default=0.01, help='lr')
parser.add_argument('--widen_factor', type=int, default=1, help='Widen factor. 4 -> 64, 8 -> 128, ...')
args = parser.parse_args()
def load_singleRes56SDNmodel(model_path):
    model = myModels4.resnetSDN(add_ic, args.interOuts)

    print(model)

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
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_singleWideRes26SDNmodel(model_path):
    # model = WideResNet(interOuts=args.interOuts)
    model = resnetSDN(num_classes=200, interOuts=args.interOuts, wideFator=wideFator)

    print(model)

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
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def trainOnePhase(model, train_loader, test_loader, criterion, outputIndex):
    best_accuracy = 0.0  # 初始化最高精度为0
    test_accuracies = onlytest(model, test_loader)
    # best_accuracy = max(test_accuracies[outputIndex-1], best_accuracy)
    # 第二阶段：固定网络主干，训练内部输出

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                          weight_decay=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)  # 每隔13个epoch将学习率衰减为原来的10分之1

    checkpoint = f'finetune/tinyImagenet/WideResnet26/finetuneCheckpoint/seed{args.seed}/wide{wideFator}/{modelOuts}/train{epochs}epochs_lr_{args.lr}'

    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    save_name = checkpoint.replace('/', '-')

    modelname = "resnet56_SDNb3b5OutputLong_w4"
    fileName = checkpoint + f"/test_accuracies.txt"
    with open("test_accuracies.txt", "w") as file:
        file.write("Test accuracies:\n")

    # if outputIndex == 2:
    #     ep = 1
    # else:
    #     ep = args.epochs
    for epoch in range(epochs):  # 假设训练10个epoch
        running_loss = 0.0

        for param in model.parameters():
            param.requires_grad = False  # 固定网络主干
        Index = 0
        countTrainOutsBn = 0
        countOtherOutsBn = 0

        for module in model.modules():
            if isinstance(module, NetworkBlock):
                for m in module.layer:
                    if isinstance(module, NetworkBlock) and m.output is not None:
                        Index += 1
                        if Index == outputIndex:
                            for name, param in m.output.layers.named_parameters():
                                # print(name)
                                param.requires_grad = True
                            for m1 in m.output.layers.modules():
                                # print(m)
                                if isinstance(m1, nn.BatchNorm2d):
                                    m1.train()
                                    countTrainOutsBn += 1
                        else:
                            for name, param in m.output.layers.named_parameters():
                                param.requires_grad = False
                                # print(name)
                            for m2 in m.output.layers.modules():
                                # print(m)
                                if isinstance(m2, nn.BatchNorm2d):
                                    m2.eval()
                                    countOtherOutsBn += 1

        # 训练当前分类头，冻结其他分类头
        countMainBn = 0
        for module in model.modules():
            if isinstance(module, NetworkBlock) and module.layer is not None:
                for m in module.layer.modules():
                    # print(m)
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        countMainBn+=1

        for module in model.modules():
            if module.module.bn1 is not None:
                m = module.module.bn1
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    break

        for i, data in enumerate(train_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            weighted_outputs = outputs[outputIndex-1] * 1
            loss = criterion(weighted_outputs, targets)  # 使用最后一个输出作为预测
            loss.requires_grad_(True)  # 加入此句就行了
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # 每100个batch打印一次损失
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        scheduler.step()  # 每个epoch结束后调用学习率衰减器的step方法

        # 测试阶段
        test_accuracies = test(model, test_loader)

        best_accList = copy.deepcopy(test_accuracies)
        best_accList[outputIndex - 1] = max(best_accList[outputIndex - 1], best_accuracy)
        print("Epoch %d, Lr:%f, test accuracies: %s, best accuracies: %s\n" % (epoch + 1, scheduler.get_last_lr()[0], test_accuracies, best_accList))
        # 将测试精度写入文件
        with open(fileName, "a") as file:
            file.write("Epoch %d, Lr:%f, test accuracies: %s, best accuracies: %s\n" % (epoch + 1, scheduler.get_last_lr()[0], test_accuracies, best_accList))

        test_accuracy = test_accuracies[outputIndex-1]
        # 更新最高精度的模型状态
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # 保存当前最高精度的模型状态
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_accuracies,
                'optimizer': optimizer.state_dict(),

            }, epoch + 1, outputIndex, best_accuracy, checkpoint=checkpoint)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
def save_checkpoint(state, epoch, outputIndex, acc, checkpoint='checkpoint'):
    filename = f'outputIndex_{outputIndex}_bestModel'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# 定义测试函数
def test(model, test_loader):
    model.eval()
    correct = [0]*nums_out
    total = [0]*nums_out
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            for i, output in enumerate(outputs):
                _, predicted = torch.max(output, 1)
                total[i] += targets.size(0)
                correct[i] += (predicted == targets).sum().item()

    accuracies = [(100 * correct[i] / total[i]) for i in range(len(correct))]
    return accuracies

def prepareData():
    train_batch = 128
    test_batch = 128
    # Data
    print('==> Preparing dataset')
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    num_classes = 200

    data_dir = './data/tiny-imagenet-200/'
    train_dataset = TinyImageNet(data_dir, train=True, transform=transform_train)
    test_dataset = TinyImageNet(data_dir, train=False, transform=transform_test)

    # 加载训练数据
    trainloader = data.DataLoader(train_dataset, batch_size=train_batch, shuffle=True, num_workers=4)
    testloader = data.DataLoader(test_dataset, batch_size=test_batch, shuffle=False, num_workers=4)

    return trainloader, testloader

def train(model, trainloader, testloader, criterion):
    # 分阶段来train

    # model_path = f'finetune/cifar10/WideResnet26/finetuneCheckpoint/{modelOuts}/train{epochs}epochs_lr_{args.lr}/outputIndex_1_bestModel'
    # model = load_singleWideRes26SDNFinetuneModel(model_path)
    for outputIndex in range(1, nums_out):
    # for outputIndex in range(nums_out-1, 0, -1):
    #     if outputIndex != 1:
        trainOnePhase(model, trainloader, testloader, criterion, outputIndex)
        model_path = f'finetune/tinyImagenet/WideResnet26/finetuneCheckpoint/seed{args.seed}/wide{wideFator}/{modelOuts}/train{epochs}epochs_lr_{args.lr}/outputIndex_{outputIndex}_bestModel'
        # model_path = f'finetune/cifar10/resnet56/finetuneCheckpoint/b2b4b6ResOut8/train120epochs/outputIndex_{outputIndex}_bestModel'
        model = load_singleWideRes26SDNFinetuneModel(model_path)

    # 多个output一起训练
    # trainOutputsTogether(model, trainloader, testloader, criterion)


def onlytest(model, testloader):
    # outputIndex = 2
    # model_path = f'finetune/cifar10/resnet56/finetuneCheckpoint_.pth.tartrainEpoch_outputIndex_{outputIndex}_bestModel'
    # model = load_singleRes56SDNFinetunemodel(model_path)
    test_accuracies = testModel(model, testloader)
    return test_accuracies

def testModel(model, testloader):
    test_accuracies = test(model, testloader)
    print('Test accuracies: %s' % (test_accuracies))
    return test_accuracies

def load_singleRes56SDNFinetunemodel(model_path):
    model = myModels4.resnetSDN(add_ic, args.interOuts)

    # print(model)

    print('==> Resuming from checkpoint..')
    try:
        checkpoint = torch.load(model_path)
    except:
        # try:
        #     ckpt_path= model_path+'/checkpoint.pth.tar'
        #     checkpoint = torch.load(ckpt_path)
        # except:
        raise Exception(f'No such file! \n {model_path}')
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def load_singleWideRes26SDNFinetuneModel(model_path):
    # model = WideResNet(interOuts=args.interOuts)
    model = resnetSDN(num_classes=200,interOuts=args.interOuts, wideFator=wideFator)

    # print(model)

    print('==> Resuming from checkpoint..')
    try:
        checkpoint = torch.load(model_path)
    except:
        # try:
        #     ckpt_path= model_path+'/checkpoint.pth.tar'
        #     checkpoint = torch.load(ckpt_path)
        # except:
        raise Exception(f'No such file! \n {model_path}')
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def getAddIc(str):
    if str == "b3b4b5b6":
        nums_out = 5
    elif str == "b1-b3":
        nums_out = 4
    return nums_out

def main():
    global nums_out, modelOuts,epochs,add_ic,newout,wideFator
    wideFator = args.widen_factor
    epochs = args.epochs
    add_ic = 0
    # 读取model
    model_path = args.model_path
    modelOuts = model_path.split('addIc_')[1]
    tmp = modelOuts
    if "NewOut" in modelOuts or "ResOut" in modelOuts:
        modelOuts = modelOuts[:-6]
    nums_out = getAddIc(modelOuts)

    newout = model_path.split('/')[-2][len('finetuneNewout'):]

    modelOuts = tmp + newout
    # path = f'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout8/addIc_b3b4b5b6ResOut'
    # model = load_singleRes56SDNFinetunemodel(path)
    # model = load_singleRes56SDNmodel(model_path)

    model = load_singleWideRes26SDNmodel(model_path)
    trainloader, testloader = prepareData()
    criterion = nn.CrossEntropyLoss()
    print("==> Start training..")
    # train earlyoutput
    # （冻结主干网络，train）
    train(model, trainloader, testloader, criterion)
    # 储存 finetune好的model
    # for _ in range(10):
    # onlytest(model, testloader)

if __name__ == '__main__':
    main()
