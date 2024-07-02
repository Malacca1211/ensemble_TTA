from tent.robustbench.model_zoo.architectures.wide_resnet_SDNDuotou import resnetSDN, NetworkBlock
from tent.robustbench.model_zoo.architectures.wide_resnet import WideResNet
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch.nn as nn
import os
import errno
import argparse

# # 复制参数
# target_net.fc1.weight.data = source_net.fc1.weight.data.clone()
# target_net.fc1.bias.data = source_net.fc1.bias.data.clone()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

parser.add_argument('--widen_factor', type=int, default=2,help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--seed', type=int, default=0, help='seed')
args = parser.parse_args()

def load_singleWideRes_model(model_path, widen_factor):
    model = WideResNet(num_classes=100, widen_factor=widen_factor)

    print(model)
    print('==> Resuming from checkpoint..')
    ckpt_path= model_path +'/model_best.pth.tar'
    try:
        checkpoint = torch.load(ckpt_path)
    except:
        try:
            # ckpt_path= model_path+'/checkpoint.pth.tar'
            checkpoint = torch.load(model_path)
        except:
            raise Exception(f'No such file! \n {ckpt_path}')

    start_epoch = checkpoint['epoch']
    model=torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def onlytest(model, testloader, nums_out):
    test_accuracies = testModel(model, testloader, nums_out)
    return test_accuracies
def test(model, test_loader, nums_out):
    model.eval()
    correct = [0]*nums_out
    total = [0]*nums_out
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            if nums_out==1:
                outputs = [outputs]
            for i, output in enumerate(outputs):
                _, predicted = torch.max(output, 1)
                total[i] += targets.size(0)
                correct[i] += (predicted == targets).sum().item()

    accuracies = [(100 * correct[i] / total[i]) for i in range(len(correct))]
    return accuracies

def testModel(model, testloader, nums_out):
    test_accuracies = test(model, testloader, nums_out)
    print('Test accuracies: %s' % (test_accuracies))
    return test_accuracies

def prepareData():
    test_batch = 128
    # Data
    print('==> Preparing dataset')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    dataloader = datasets.CIFAR100
    num_classes = 100


    testset = dataloader(root='./data', train=False,
                         download=False, transform=transform_test)
    testloader = data.DataLoader(
        testset, batch_size=test_batch, shuffle=False, num_workers=4)

    return testloader

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
def save_checkpoint(state, checkpoint='checkpoint', filename='model_best.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def main():
    seed = args.seed
    widen_factor = args.widen_factor

    model_path = f'checkpoint/cifar100/wideResnet_SDN_w{widen_factor}/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_{seed}/DUAN2DUANResout/wide{widen_factor}'
    checkpointPath = f'checkpoint/cifar100/wideResnet_SDN_w{widen_factor}/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_{seed}/finetuneResout0/Wide{widen_factor}/addIc_b1-b3ResOut/'

    if not os.path.isdir(checkpointPath):
        mkdir_p(checkpointPath)

    model1 = load_singleWideRes_model(model_path, widen_factor)

    model2 = resnetSDN(num_classes=100, interOuts=0, wideFator=widen_factor)
    model1.cuda()
    model2.cuda()
    testloader = prepareData()

    c1, c2 = 0, 0
    d1, d2 = 0, 0

    for name2, param2 in model1.named_parameters():
        print(f"Model1 Parameter name: {name2}\tShape: {param2.shape}")
        d1+=1

    for name2, param2 in model2.named_parameters():
        print(f"Model2 Parameter name: {name2}\tShape: {param2.shape}")
        d2+=1

    print(f"Total model1 params: {d1}\tTotal model2 params: {d2}")

    for name2, param2 in model2.named_parameters():
        # print(f"Model2 Parameter name: {name2}\tShape: {param2.shape}")
        for name1, param1 in model1.named_parameters():
            if name2 in name1 and param2.shape == param1.shape:
                param2.data = param1.data.clone()
                c1+=1
                break

    bn_layers1 = []
    for name, module in model1.named_modules():
        if isinstance(module, (nn.BatchNorm2d)):
            bn_layers1.append((name, module))

    bn_layers2 = []
    for name, module in model2.named_modules():
        if isinstance(module, (nn.BatchNorm2d)):
            bn_layers2.append((name, module))

    for idx in range(len(bn_layers1)):
        bn_layer1 = bn_layers1[idx][1]
        bn_layer2 = bn_layers2[idx][1]
        bn_layer2.weight.data = bn_layer1.weight.data.clone()
        bn_layer2.bias.data = bn_layer1.bias.data.clone()
        bn_layer2.running_var.data = bn_layer1.running_var.data.clone()
        bn_layer2.running_mean.data = bn_layer1.running_mean.data.clone()
        c2+=1

    print(f"Total model1 params: {c1}\tTotal model2 params: {c2}")

    test_accuracies = onlytest(model1, testloader, 1)
    onlytest(model2, testloader, 4)

    model2 = torch.nn.DataParallel(model2)
    save_checkpoint({
        'state_dict': model2.state_dict(),
        'acc': test_accuracies,

    }, checkpoint=checkpointPath)

if __name__ == '__main__':
    main()