import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size/4)
    else:
        return -1

def feature_reduction_formula2(input_feature_map_size):
    if input_feature_map_size >= 8:
        return int(input_feature_map_size/8)
    else:
        return -1
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, add_output, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        if out_planes == 16 * wideFatorg:
            input_size = 32#64 if gnum_classes == 200 else 32
        elif out_planes == 32 * wideFatorg:
            input_size = 16#32 if gnum_classes == 200 else 16
        elif out_planes == 64 * wideFatorg:
            input_size = 8#16 if gnum_classes == 200 else 8
        if add_output:
            self.output = InternalClassifier(input_size, out_planes, gnum_classes)
            self.has_output = True

        else:
            self.output = None
            self.has_output = False
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        fwd = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        if self.has_output:
            return fwd, self.has_output, self.output(fwd)
        else:
            return fwd, self.has_output, None

class InternalClassifier(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
        super(InternalClassifier, self).__init__()
        #red_kernel_size = -1 # to test the effects of the feature reduction
        red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        self.output_channels = output_channels

        if red_kernel_size == -1:
            self.linear = nn.Linear(output_channels*input_size*input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size/red_kernel_size)

            layers = nn.ModuleList()
            conv_layers = []

            for i in range(gInterOuts):
                conv_layer = []
                conv_layer.append(
                    nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False))
                conv_layer.append(nn.BatchNorm2d(output_channels))
                conv_layer.append(nn.ReLU())

                # 将两层卷积连接在一起
                if i % 2 == 0:
                    conv_layers.append(nn.Sequential(*conv_layer))
                else:
                    # 添加残差连接
                    conv_layers[-1] = nn.Sequential(conv_layers[-1], nn.Sequential(*conv_layer))

            # 添加到主网络层
            for conv_layer in conv_layers:
                layers.append(conv_layer)
                layers.append(nn.ReLU())

            layers = nn.Sequential(*layers)

            # self.alpha = nn.Parameter(torch.rand(1))
            self.alpha = nn.Parameter(torch.full((1,), 0.5))

            layers.append(nn.MaxPool2d(kernel_size=red_kernel_size))
            layers.append(nn.AvgPool2d(kernel_size=red_kernel_size))

            linear = nn.Linear(output_channels*red_input_size*red_input_size, num_classes)
            layers.append(linear)

            self.forward = self.forward_w_pooling
        self.layers = layers

    def forward_w_pooling(self, x):
        # 分割layers
        residual_layers = self.layers[:-3]  # 前面部分
        output_layers = self.layers[-3:]  # 后面部分
        out = x
        for layer in residual_layers:
            out = layer(out)

        avgp = self.alpha*output_layers[0](out)
        maxp = (1 - self.alpha)*output_layers[1](out)
        mixed = avgp + maxp

        x3 = self.layers[-1](mixed.reshape(mixed.size(0), -1))
        return x3

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, add_output, add_icList, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, add_icList, stride, dropRate)

        # if add_output:
        #     self.output = InternalClassifier(input_size, out_planes, 10)
        #     self.has_output = True
        #
        # else:
        #     self.output = None
        #     self.has_output = False


    def _make_layer(self, block, in_planes, out_planes, nb_layers, add_icList, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, add_icList[i]==1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        myOutput = []
        myHas_output = False
        fwd = x
        for layer in self.layer:
            fwd, has_output, output = layer(fwd)
            if has_output:
                myOutput.append(output)
                myHas_output = True
        return fwd, myHas_output, myOutput

class WideResNet(nn.Module):
    """ Based on code from https://github.com/yaodongyu/TRADES """
    def __init__(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, interOuts=0, add_ic=[], dropRate=0.0, bias_last=True):
        super(WideResNet, self).__init__()
        global gInterOuts, gnum_classes
        gnum_classes = num_classes
        gInterOuts = interOuts
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, False, add_ic[0], dropRate)
        if sub_block1:
            # 1st sub-block
            self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, True, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, False, add_ic[1], dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, False, add_ic[2], dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not m.bias is None:
                m.bias.data.zero_()

    def forward(self, x):
        fwd = self.conv1(x)
        outputs = []
        blocks = [self.block1, self.block2, self.block3]

        for layer in blocks:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.extend(output)

        out = self.relu(self.bn1(fwd))
        poolKernel = 8#16 if gnum_classes == 200 else 8
        out = F.avg_pool2d(out, poolKernel)
        # out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        lastOutput = self.fc(out)
        outputs.append(lastOutput)
        return outputs

def resnetSDN(num_classes, interOuts,wideFator):
    # add_ic = [[0,0,1,0],[0,1,0,0],[1,0,0,0]]
    # add_ic = [[1,1,1,1],[1,1,1,1],[1,1,1,0]]
    add_ic = [[1,0,0,0],[1,0,0,0],[1,0,0,0]]
    # add_ic = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    global wideFatorg
    wideFatorg = wideFator

    model = WideResNet(num_classes=num_classes, widen_factor=wideFator,add_ic=add_ic, interOuts=interOuts)
    print(model)
    return model