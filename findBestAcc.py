
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import utils.myModels5 as myModels5
import utils.findBestModel as findBestModel
import torch.nn as nn
import os
import errno
import argparse
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-m', '--model_path', default='checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneZuhe/addIc_b1b2b3b4b5b6', type=str)
parser.add_argument('--lr', type=float, default=0.001, help='lr')
parser.add_argument('--step', type=int, default=20, help='step')
parser.add_argument('--nums_layers1', type=int, default=8, help='nums_layers')
parser.add_argument('--nums_layers2', type=int, default=8, help='nums_layers')
parser.add_argument('--nums_layers3', type=int, default=8, help='nums_layers')
parser.add_argument('--nums_layers4', type=int, default=8, help='nums_layers')
parser.add_argument('--nums_layers5', type=int, default=8, help='nums_layers')
parser.add_argument('--nums_layers6', type=int, default=8, help='nums_layers')
parser.add_argument('--epochs', type=int, default=60, help='epochs')

parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
args = parser.parse_args()


def load_singleRes56SDNmodel(model_path):
    model = myModels5.resnetSDN(add_ic)

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
    start_epoch = checkpoint['epoch']
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    return model

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

def prepareData():
    train_batch = 128
    test_batch = 128
    # Data
    print('==> Preparing dataset')
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
    dataloader = datasets.CIFAR10
    num_classes = 10

    trainset = dataloader(root='./data', train=True,
                          download=True, transform=transform_train)
    trainloader = data.DataLoader(
        trainset, batch_size=train_batch, shuffle=True, num_workers=4)

    testset = dataloader(root='./data', train=False,
                         download=False, transform=transform_test)
    testloader = data.DataLoader(
        testset, batch_size=test_batch, shuffle=False, num_workers=4)

    return trainloader, testloader


def train(mainModel, trainModels, trainloader, testloader, criterion):
    # mainModel加载并推理

    # 训练新NET
    best_accuracy = 0.0  # 初始化最高精度为0
    # 第二阶段：固定网络主干，训练内部输出
    lr = args.lr
    step_size = args.step
    gamma = args.gamma

    optimizers = []
    for idx in range(len(trainModels)):
        optimizers.append(optim.SGD(params=trainModels[idx].parameters(), lr=lr, momentum=0.9, weight_decay=0.0001))
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # 每隔13个epoch将学习率衰减为原来的10分之1

    checkpoint = f'finetune/cifar10/resnet56/finetuneZuhe/nums_output{len(trainModels)}/lr_{lr}step_{step_size}gamma_{gamma}epochs_{epochs}'

    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    save_name = checkpoint.replace('/', '-')

    fileName = checkpoint + f"/test_accuracies.txt"
    with open("test_accuracies.txt", "w") as file:
        file.write("Test accuracies:\n")


    for epoch in range(epochs):  # 假设训练10个epoch
        running_loss = [0.0] * len(trainModels)
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            for optimizer in optimizers:
               optimizer.zero_grad()
            netInputs = mainModel(inputs)

            # define NET 注意输入维度
            for idx in range(len(trainModels)):
                outputs = trainModels[idx](netInputs[idx])

                loss = criterion(outputs, targets)  # 使用最后一个输出作为预测
                loss.requires_grad_(True)  # 加入此句就行了
                loss.backward(retain_graph=True)
                optimizers[idx].step()
                running_loss[idx] += loss.item()
            if i % 100 == 99:  # 每100个batch打印一次损失
                print('[%d, %5d]' %(epoch + 1, i + 1))
                for i, loss in enumerate(running_loss):
                    print('loss{}: {:.3f}'.format(i + 1, loss / 100))
                running_loss = [0.0] * len(trainModels)
        # scheduler.step()  # 每个epoch结束后调用学习率衰减器的step方法

        # 测试阶段
        test_accuracies = test(mainModel, trainModels, testloader)
        print('Epoch %d, test accuracies: %s' % (epoch + 1, test_accuracies))
        # # 将测试精度写入文件
        with open(fileName, "a") as file:
            file.write("Epoch %d, test accuracies: %s\n" % (epoch + 1, test_accuracies))

def test(mainModel, testModels, test_loader):
    for testModel in testModels:
        testModel.eval()
    correct = [0] * len(testModels)
    total = [0] * len(testModels)
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            netInputs = mainModel(inputs)

            for idx in range(len(testModels)):
                output = testModels[idx](netInputs[idx])
                _, predicted = torch.max(output, 1)
                total[idx] += targets.size(0)
                correct[idx] += (predicted == targets).sum().item()
    accuracies = []
    for idx in range(len(testModels)):
        accuracies.append(100 * correct[idx] / total[idx])
    return accuracies

def main():
    global nums_out, modelOuts,epochs,add_ic,newout
    epochs = args.epochs

    add_ic = [
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0]
    ]

    # 读取model
    model_path = args.model_path
    mainModel = load_singleRes56SDNmodel(model_path)
    trainloader, testloader = prepareData()
    criterion = nn.CrossEntropyLoss()
    print("==> Start training..")
    # train earlyoutput
    # （冻结主干网络，train）
    interOutput1 = findBestModel.InteralOutput(input_size=32, output_channels=16, num_classes=10, nums_layers=args.nums_layers1).cuda()
    interOutput2 = findBestModel.InteralOutput(input_size=32, output_channels=16, num_classes=10, nums_layers=args.nums_layers2).cuda()
    interOutput3 = findBestModel.InteralOutput2(input_size=32, output_channels=32, num_classes=10, nums_layers=args.nums_layers3).cuda()
    interOutput4 = findBestModel.InteralOutput2(input_size=32, output_channels=32, num_classes=10, nums_layers=args.nums_layers4).cuda()
    interOutput5 = findBestModel.InteralOutput3(input_size=32, output_channels=64, num_classes=10, nums_layers=args.nums_layers5).cuda()
    interOutput6 = findBestModel.InteralOutput3(input_size=32, output_channels=64, num_classes=10, nums_layers=args.nums_layers6).cuda()

    # interOutputs = [interOutput5, interOutput6]

    interOutputs = [interOutput1, interOutput2, interOutput3, interOutput4, interOutput5, interOutput6]

    train(mainModel, interOutputs, trainloader, testloader, criterion)
    # 储存 finetune好的model

    # onlytest(model, testloader)

if __name__ == '__main__':
    main()
