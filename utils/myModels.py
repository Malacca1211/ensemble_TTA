import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, conv1x1, conv3x3
import torchvision.models as models

class ResNet56(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0):
        super(ResNet56, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 9, stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(BasicBlock, 32, 9, stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(BasicBlock, 64, 9, stride=2, dropout_prob=dropout_prob)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def _make_layer(self, block, planes, blocks, stride=1, dropout_prob=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet32NewEarlyOutput(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0):
        super(ResNet32NewEarlyOutput, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 5, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 5, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 5, stride=2)
        self.avgpool1 = nn.AdaptiveAvgPool2d((8, 8))
        self.avgpool2 = nn.AdaptiveAvgPool2d((4, 4))
        self.avgpool3 = nn.AdaptiveAvgPool2d((2, 2))

        # 第一层卷积
        self.convO11 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 第二层卷积
        self.convO12 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 第三层卷积
        self.convO13 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(64 * BasicBlock.expansion * 4, num_classes)
        self.fc2 = nn.Linear(64 * BasicBlock.expansion, num_classes)

        # 添加额外的输出层，并调整维度
        self.output_layer1 = nn.Linear(16*8*8*4, 64 * BasicBlock.expansion)
        self.output_layer2 = nn.Linear(32*4*4*2, 64 * BasicBlock.expansion)
        self.dropout = nn.Dropout(dropout_prob)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        output1 = self.layer1(x)
        output2 = self.layer2(output1)
        x = self.layer3(output2)
        x = self.avgpool3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # 返回主要输出以及额外输出
        output1 = self.avgpool1(output1)
        output1 = F.relu(self.convO11(output1))
        output1 = F.relu(self.convO12(output1))
        output1 = F.relu(self.convO13(output1))
        output1 = torch.flatten(output1, 1)

        output2 = self.avgpool2(output2)
        output2 = F.relu(self.convO12(output2))
        output2 = F.relu(self.convO13(output2))
        output2 = torch.flatten(output2, 1)

        output1 = F.relu(self.output_layer1(output1))
        output2 = F.relu(self.output_layer2(output2))

        output1 = self.fc2(output1)
        output2 = self.fc2(output2)

        return x, output1, output2
class ResNet56EarlyOutput(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0.5):
        super(ResNet56EarlyOutput, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 5, stride=1) #9
        self.layer2 = self._make_layer(BasicBlock, 32, 5, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 5, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)
        # 添加额外的输出层，并调整维度
        self.output_layer1 = nn.Linear(16, 64 * BasicBlock.expansion)
        self.output_layer2 = nn.Linear(32, 64 * BasicBlock.expansion)
        self.output_layer3 = nn.Linear(64, 64 * BasicBlock.expansion)
        self.dropout = nn.Dropout(dropout_prob)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        output1 = self.layer1(x)
        x = self.dropout(x)
        output2 = self.layer2(output1)
        x = self.dropout(x)
        output3 = self.layer3(output2)
        x = self.dropout(x)
        x = self.avgpool(output3)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # 返回主要输出以及额外输出
        output1 = self.avgpool(output1)
        output1 = torch.flatten(output1, 1)
        output2 = self.avgpool(output2)
        output2 = torch.flatten(output2, 1)
        output3 = self.avgpool(output3)
        output3 = torch.flatten(output3, 1)

        output1 = self.output_layer1(output1)
        output2 = self.output_layer2(output2)
        output3 = self.output_layer3(output3)

        output1 = self.fc(output1)
        output2 = self.fc(output2)
        output3 = self.fc(output3)

        return x, output1, output2, output3


def resnet56_with_dropout(num_classes=10, dropout_prob=0.2):
    return ResNet56(num_classes=num_classes, dropout_prob=dropout_prob)

def resnet56Early_with_dropout(num_classes=10, dropout_prob=0.1):
    return ResNet56EarlyOutput(num_classes=num_classes, dropout_prob=dropout_prob)

def resnet32NewEarly_with_dropout(num_classes=10, dropout_prob=0.1):
    return ResNet32NewEarlyOutput(num_classes=num_classes, dropout_prob=dropout_prob)

class ResNetWithDropout(nn.Module):
    def __init__(self, num_classes=1000, dropout_prob=0.1):
        super(ResNetWithDropout, self).__init__()
        resnet = models.resnet34(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # 去除最后的全连接层和平均池化层
        # 添加一个新的全连接层
        self.fc1 = nn.Linear(512, 256)
        # 添加dropout层
        self.dropout = nn.Dropout(dropout_prob)
        # 最后的线性分类器
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.dropout(x)  # 在全连接层之前应用Dropout
        # x = self.fc(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x