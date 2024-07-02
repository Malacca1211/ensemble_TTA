import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size/4)
    else:
        return -1

class InteralOutput(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, nums_layers, alpha=0.5):
        super(InteralOutput, self).__init__()
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

            for i in range(nums_layers):
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

            self.alpha = nn.Parameter(torch.rand(1))
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

        x3 = output_layers[2](mixed.reshape(mixed.size(0), -1))
        return x3

class InteralOutput2(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, nums_layers, alpha=0.5):
        super(InteralOutput2, self).__init__()
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

            for i in range(nums_layers):
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

            self.alpha = nn.Parameter(torch.rand(1))
            layers.append(nn.MaxPool2d(kernel_size=red_kernel_size))
            layers.append(nn.AvgPool2d(kernel_size=red_kernel_size))

            linear = nn.Linear(128, num_classes)
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

        x3 = output_layers[2](mixed.reshape(mixed.size(0), -1))
        return x3

class InteralOutput3(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, nums_layers, alpha=0.5):
        super(InteralOutput3, self).__init__()
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

            for i in range(nums_layers):
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

            self.alpha = nn.Parameter(torch.rand(1))
            layers.append(nn.MaxPool2d(kernel_size=red_kernel_size))
            layers.append(nn.AvgPool2d(kernel_size=red_kernel_size))

            linear = nn.Linear(64, num_classes)
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

        x3 = output_layers[2](mixed.reshape(mixed.size(0), -1))
        return x3