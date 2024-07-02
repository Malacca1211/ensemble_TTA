import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.configs import InfoPro, InfoPro_balanced_memory

from networks.auxiliary_nets import Decoder, AuxClassifier

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
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

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class InfoProWideResNet(nn.Module):
    """ Based on code from https://github.com/yaodongyu/TRADES """
    def __init__(self, arch,depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True ,local_module_num=8, batch_size=128, image_size=32,
                 balanced_memory=False, dataset='cifar10', 
                 wide_list=(16, 16, 32, 64), 
                 aux_net_config='1c2f', local_loss_mode='contrast',
                 aux_net_widen=1, aux_net_feature_dim=128):
        super(InfoProWideResNet, self).__init__()

        assert arch in ['wideresnet28', 'wideresnet40'], "This repo supports resnet32 and resnet110 currently. " \
                                                  "For other networks, please set network configs in .configs."


        self.inplanes = wide_list[0]
        self.dropout_rate = dropRate,
        self.feature_num = wide_list[-1]
        self.class_num = num_classes
        self.local_module_num = local_module_num
        # self.layers = layers
       
        
        
        
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        self.block_ly_num = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(self.block_ly_num, nChannels[0], nChannels[1], block, 1, dropRate)
        if sub_block1:
            # 1st sub-block
            self.sub_block1 = NetworkBlock(self.block_ly_num, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(self.block_ly_num, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(self.block_ly_num, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]


        try:
            self.infopro_config = InfoPro_balanced_memory[arch][dataset][local_module_num] \
                if balanced_memory else InfoPro[arch][local_module_num]
        except:
            raise NotImplementedError

        for item in self.infopro_config:
            module_index, layer_index = item

            exec('self.decoder_' + str(module_index) + '_' + str(layer_index) +
                 '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')

            exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) +
                 '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                 'loss_mode=local_loss_mode, class_num=class_num, '
                 'widen=aux_net_widen, feature_dim=aux_net_feature_dim)')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not m.bias is None:
                m.bias.data.zero_()

        if 'cifar' in dataset:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
        else:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).cuda()

    def _image_restore(self, normalized_image):
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]


    def forward_original(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def forward(self, img, target=None,
                ixx_1=0, ixy_1=0,
                ixx_2=0, ixy_2=0):

        if self.training:
            stage_i = 0
            layer_i = 0
            local_module_i = 0
        
            x = self.conv1(img)
            # x = self.bn1(x)
            # x = self.relu(x)
        
            if local_module_i <= self.local_module_num - 2:
                if self.infopro_config[local_module_i][0] == stage_i \
                        and self.infopro_config[local_module_i][1] == layer_i:
                    ratio = local_module_i / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                    ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                    ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                    loss_ixx = eval('self.decoder_' + str(stage_i) + '_' + str(layer_i))(x, self._image_restore(img))
                    loss_ixy = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i))(x, target)
                    loss = ixx_r * loss_ixx + ixy_r * loss_ixy
                    loss.backward()
                    x = x.detach()
                    local_module_i += 1
        
            for stage_i in (1, 2, 3): #stage_i :usd to
                for layer_i in range(self.block_ly_num):
                    x = eval('self.block' + str(stage_i))[layer_i](x)
        
                    if local_module_i <= self.local_module_num - 2:
                        if self.infopro_config[local_module_i][0] == stage_i \
                                and self.infopro_config[local_module_i][1] == layer_i:
                            ratio = local_module_i / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                            ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                            ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                            # loss_ixx: reconstruction loss?
                            # img: 1024*3*32*32
                            # x: 1024*32*16*16
                            loss_ixx = eval('self.decoder_' + str(stage_i) + '_' + str(layer_i))(x, self._image_restore(img))
                            loss_ixy = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i))(x, target)
                            loss = ixx_r * loss_ixx + ixy_r * loss_ixy
                            loss.backward()
                            x = x.detach()
                            # loss.backward()
                            local_module_i += 1
        
            x = self.relu(self.bn1(x))
            x = F.avg_pool2d(x, 8)
            x = x.view(-1, self.nChannels)

            logits = self.fc(x)
            loss = self.criterion_ce(logits, target)
            loss.backward()
            return logits, loss
        
        else:
            out = self.conv1(x)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.nChannels)
            logits=self.fc(out)
            loss = self.criterion_ce(logits, target)
            return logits, loss

