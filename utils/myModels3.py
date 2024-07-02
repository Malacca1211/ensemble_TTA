import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# the formula for feature reduction in the internal classifiers
def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size/4)
    else:
        return -1

class BasicBlockWOutput(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, params, stride=1):
        super(BasicBlockWOutput, self).__init__()
        add_output = params[0]
        num_classes = params[1]
        input_size = params[2]
        self.output_id = params[3]

        self.depth = 2

        layers = nn.ModuleList()

        conv_layer = []
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU())
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))

        layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * channels)
            )

        layers.append(shortcut)
        layers.append(nn.ReLU())

        self.layers = layers

        if add_output:
            self.output = None #InternalClassifier(input_size, self.expansion * channels, num_classes)
            self.no_output = False

        else:
            self.output = None
            self.forward = self.only_forward
            self.no_output = True

    def forward(self, x):
        fwd = self.layers[0](x)  # conv layers
        fwd = fwd + self.layers[1](x)  # shortcut
        return self.layers[2](fwd), 1, self.output(fwd)  # output layers for this module

    def only_output(self, x):
        fwd = self.layers[0](x)  # conv layers
        fwd = fwd + self.layers[1](x)  # shortcut
        fwd = self.layers[2](fwd)  # activation
        out = self.output(fwd)  # output layers for this module
        return out

    def only_forward(self, x):
        fwd = self.layers[0](x)  # conv layers
        fwd = fwd + self.layers[1](x)  # shortcut
        return self.layers[2](fwd), 0, None  # activation

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
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.linear = nn.Linear(output_channels*red_input_size*red_input_size, num_classes)
            self.forward = self.forward_w_pooling

    def forward_w_pooling(self, x):
        avgp = self.alpha*self.max_pool(x)
        maxp = (1 - self.alpha)*self.avg_pool(x)
        mixed = avgp + maxp
        # return self.linear(mixed.view(mixed.size(0), -1))
        return self.linear(mixed.reshape(mixed.size(0), -1))

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))

class InternalClassifier2(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
        super(InternalClassifier2, self).__init__()
        #red_kernel_size = -1 # to test the effects of the feature reduction
        red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        self.output_channels = output_channels

        if red_kernel_size == -1:
            self.linear = nn.Linear(output_channels*input_size*input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size/red_kernel_size)

            layers = nn.ModuleList()
            channels1 = 16
            channels2 = 8
            conv_layer = []
            conv_layer.append(nn.Conv2d(output_channels, channels1, kernel_size=3, stride=1, padding=1, bias=False))
            conv_layer.append(nn.BatchNorm2d(channels1))
            conv_layer.append(nn.ReLU())
            conv_layer.append(nn.Conv2d(channels1, channels2, kernel_size=3, stride=1, padding=1, bias=False))
            conv_layer.append(nn.BatchNorm2d(channels2))

            layers.append(nn.Sequential(*conv_layer))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=red_kernel_size))
            # layers.append(nn.AvgPool2d(kernel_size=red_kernel_size))

            linear = nn.Linear(channels2*red_input_size*red_input_size, num_classes)

            layers.append(linear)

            self.forward = self.forward_w_pooling
        self.layers = layers

    def forward_w_pooling(self, x):
        x1 = self.layers[0](x)
        x2 = self.layers[1](x1)
        x3 = self.layers[2](x2)
        x4 = self.layers[3](x3.reshape(x3.size(0), -1))
        # x4 = self.layers[3](x3.view(x3.size(0), -1))
        return x4

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))

class ResNet_SDN(nn.Module):
    def __init__(self, params):
        super(ResNet_SDN, self).__init__()
        self.num_blocks = params['num_blocks']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.coeffs = params['max_coeffs']
        self.block_type = params['block_type']
        self.add_out_nonflat = params['add_ic']
        self.add_output = [item for sublist in self.add_out_nonflat for item in sublist]
        self.init_weights = params['init_weights']
        self.in_channels = 16
        self.num_output = sum(self.add_output) + 1

        self.init_depth = 1
        self.end_depth = 1
        self.cur_output_id = 0

        self.block = BasicBlockWOutput

        init_conv = []

        if self.input_size == 32:  # cifar10
            self.cur_input_size = self.input_size
            init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        else:  # tiny imagenet
            self.cur_input_size = int(self.input_size / 2)
            init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False))

        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU())

        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layer(self.in_channels, block_id=0, stride=1))

        self.cur_input_size = int(self.cur_input_size / 2)
        self.layers.extend(self._make_layer(32, block_id=1, stride=2))

        self.cur_input_size = int(self.cur_input_size / 2)
        self.layers.extend(self._make_layer(64, block_id=2, stride=2))

        end_layers = []

        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(Flatten())
        end_layers.append(nn.Linear(64 * self.block.expansion, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def _make_layer(self, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for cur_block_id, stride in enumerate(strides):
            add_output = self.add_out_nonflat[block_id][cur_block_id]
            params = (add_output, self.num_classes, int(self.cur_input_size), self.cur_output_id)
            layers.append(self.block(self.in_channels, channels, params, stride))
            self.in_channels = channels * self.block.expansion
            self.cur_output_id += add_output

        return layers

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = []
        fwd = self.init_conv(x)
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)

        return outputs

    # takes a single input
    def early_exit(self, x):
        confidences = []
        outputs = []

        fwd = self.init_conv(x)
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)

            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)

                confidence = torch.max(softmax)
                confidences.append(confidence)

                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output, output_id, is_early

                output_id += is_output

        output = self.end_layers(fwd)
        outputs.append(output)

        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        confidences.append(confidence)
        # max_confidence_output = np.argmax(confidences)
        confidences_tensor = torch.tensor(confidences)
        max_confidence_output = np.argmax(confidences_tensor.cpu().detach().numpy())
        is_early = False
        return outputs[max_confidence_output], max_confidence_output, is_early

def resnetSDN(add_ic=[]):
    config = {
        'task': 'cifar10',
        'input_size': 32,
        'num_classes': 10,
        'block_type': 'basic',
        'num_blocks': [9, 9, 9],
        'add_ic': add_ic,
        'max_coeffs': [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1],
        'network_type': 'resnet56',
        'augment_training': True,
        'init_weights': True,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'milestones': [35, 60, 85],
        'gammas': [0.1, 0.1, 0.1],
        'architecture': 'sdn',
        'converted_from': 'cifar10_resnet56_cnn',
        'optimizer': 'SGD'
    }
    model = ResNet_SDN(config)
    print(model)
    return model

# initial
# 'add_ic': [
#     [0, 0, 0, 1, 0, 0, 0, 1, 0],
#     [0, 0, 1, 0, 0, 0, 1, 0, 0],
#     [0, 1, 0, 0, 0, 1, 0, 0, 0]
# ],

# resnetSDN()