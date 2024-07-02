import numpy as np
import matplotlib.pyplot as plt
import torchvision
import sys
import torch
import torch.nn
from yacs.config import CfgNode as CfgNode
import os
import json
import re


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

def save_tensor(x:torch.tensor, batch_idx,path):
    '''save tensor to path, note that x is a batch of images, b c h w'''
    import matplotlib.pyplot as plt
    if len(x.shape)==4:
        try:
            plt.imshow(x.permute(0,2,3,1).cpu().numpy()[batch_idx])
        except:
            plt.imshow(x.permute(0,2,3,1).detach().cpu().numpy()[batch_idx])
    elif len(x.shape)==3:
        try:
            plt.imshow(x.cpu().numpy()[batch_idx])
        except:
            plt.imshow(x.detach().cpu().numpy()[batch_idx])
    else:
        raise ValueError('x must be 3 or 4 dim')
    
    plt.savefig(path)
    


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def feature_imshow(inp, title=None, normalize=False):
    """Imshow for Tensor.
    """

    if normalize:
        inp = normalize_feature_map(inp)

    inp = inp.cpu().detach().numpy()  # .transpose((1, 2, 0))

    # inp = np.clip(inp, 0, 1)
    concatenated_features = concat_fms(inp)
    plt.imshow(concatenated_features, cmap='gray')
    plt.savefig(title)


def normalize_feature_map(feature_map):
    '''normalize featuremaps'''
    # Get mean and standard deviation
    mean = feature_map.mean()
    std = feature_map.std()

    # Normalize feature map
    feature_map_norm = (feature_map - mean) / std
    feature_map_norm = feature_map_norm * 0.5 + 0.5  # Scale to range [0, 1]
    return feature_map_norm


def concat_fms(features):
    num_channels = features.shape[0]
    imsize = features.shape[1]
    features = features.reshape(num_channels, 1, imsize, imsize)
    # 计算拼接后的正方形大图的大小
    square_size = int(np.ceil(np.sqrt(num_channels)))
    # 初始化大图的数组
    big_img = np.zeros((square_size*imsize, square_size*imsize))

    # 拼接特征图
    for i in range(num_channels):
        # 计算特征图在大图中的位置
        row = (i // square_size) * imsize
        col = (i % square_size) * imsize

        # 把特征图复制到大图中对应的位置
        big_img[row:row+imsize, col:col+imsize] = features[i, 0]
    return big_img


def cal_layer_loss(last_state, state):
    # Initiate losses for each layer
    losses = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    criterian = torch.nn.MSELoss(reduction = 'sum')
    sizes = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for nm in state.keys():
        if nm.startswith('module.layer'):
            for ln in range(1, 4):  # Resnet has 3 layers in total
                if nm.startswith('module.layer'+str(ln)) and (('weight' in nm) or ('bias' in nm)):
                    losses[ln].append(
                        criterian(state[nm], last_state[nm]).cpu().numpy())
                    sizes[ln] += sum(state[nm].size())
                    break

        elif ('fc' in nm):  # module after layer3 (only fc
            losses[4].append(
                criterian(state[nm], last_state[nm]).cpu().numpy())
            sizes[4] += sum(state[nm].size())
        else:
            if ('weight' in nm) or ('bias' in nm):  # modules before layer1
                losses[0].append(
                    criterian(state[nm], last_state[nm]).cpu().numpy())
                sizes[0] += sum(state[nm].size())
    # mean difference
    losses_ = [sum(losses[i])/sizes[i] for i in range(5)]
    return losses_

def plot_module_loss(args):
    fn = args.checkpoint+'/log_ly.txt'
    with open(fn, 'r') as f:
        lines = f.readlines()
    keys = lines[0].strip('\n').strip('\t').split('\t')
    losses = dict.fromkeys(keys)
    for n in keys:
        losses[n] = []
    for n in lines[1:-1]:
        n = n.strip('\n').strip('\t')
        for i in range(len(keys)):
            losses[keys[i]].append(float(n.split('\t')[i]))

    for n in keys:
        plt.plot(losses[n], label=n)
        plt.ylim([0, 0.002])
        plt.legend()
        plt.title(n)
        plt.savefig(
            f'{args.checkpoint}/{n}.png')
        # plt.show()

def write_json(var_list,file_name):
    with open(file_name, 'w', encoding='UTF-8') as fp:
        try:
            fp.write(json.dumps(var_list, indent=2, ensure_ascii=False))
        except:
            x_dict = []
            for item in var_list:
                if isinstance(item, torch.Tensor):
                    item = item.cpu().detach().numpy().tolist()
                if isinstance(item,np.ndarray):
                    item=item.tolist()
                x_dict.append(item)
            fp.write(json.dumps(x_dict, indent=2, ensure_ascii=False))      

        

def writefile(filepath, filename):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filewrite = open(os.path.join(filepath,filename) , 'a')
    return filewrite

def visualize_mnistc_dataset(data_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    corrup_names = os.listdir(data_dir)
    fms = []
    legends = []
    for n in corrup_names:
        fp = os.path.join(data_dir,n,'train_images.npy')
        imgs = np.load(fp)
        plt.imshow(imgs[0])
        fms.append(imgs[0])
        legends.append(n)
        save_dir = os.path.dirname(data_dir)+'/vis_mnistc'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # plt.savefig(os.path.join(save_dir,n+'.png'))
    # plot all in a subplot, four lines:
    plt.figure()
    num_images = len(fms)
    num_rows = int(np.sqrt(num_images))
    num_cols = int(np.ceil(num_images / num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(fms[i])
        plt.title(legends[i])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'all.png'))

def getLrAndDp(path):
    lr_match = re.search(r'lr(\d+\.\d+)', path)
    drop_match = re.search(r'drop(\d+\.\d+)', path)
    lr_value = float(lr_match.group(1))
    drop_value = float(drop_match.group(1))
    return lr_value, drop_value
def save_EarlyoutModel_and_error(error, modelCount, modelPath, InferDp):
    lr, dp = getLrAndDp(modelPath)

    if 'EarlyOutput' in modelPath:
        file_path = f"testRecord/EarlyOutput_and_error_Model{modelCount}.txt"
    else:
        file_path = f"testRecord/Normal_and_error_Model{modelCount}.txt"
    try:
        # 尝试打开文件，如果文件不存在则创建文件
        with open(file_path, "x") as file:
            file.write(f"Lr: {lr}  ")
            file.write(f"dp: {dp}  ")
            file.write(f"InferDp: {InferDp}  ")
            file.write(f"Error: {error}\n")
            print("File created and data saved successfully!")
    except FileExistsError:
        # 如果文件已经存在，则追加内容
        with open(file_path, "a") as file:
            file.write(f"\nLr: {lr}  ")
            file.write(f"dp: {dp}  ")
            file.write(f"InferDp: {InferDp}  ")
            file.write(f"Error: {error}\n")
            print("Data appended to existing file successfully!")

def fromConfidenceGetData(output, label, threshold):
    # 计算每行最大值与第二大值的差异和对应的索引
    diff_values, indices = output.topk(k=2, dim=1)

    # 计算差异值
    max_diff_values = diff_values[:, 0]
    second_max_diff_values = diff_values[:, 1]
    diff_values = max_diff_values - second_max_diff_values

    # 找到 diff_values 大于 10 的索引
    high_diff_indices = torch.nonzero(diff_values > threshold).squeeze()

    # 根据索引提取新的 output 和 label
    new_output = output[high_diff_indices]
    new_label = label[high_diff_indices]

    # print("大于10的差异值的坐标：", high_diff_indices)
    # print("对应的新的 output：", new_output)
    # print("对应的新的 label：", new_label)
    return new_output, new_label

def calculate_confidence(output, label, corruption_type):
    print(111)
    # output = torch.cat(outputAll, dim=0)
    # 计算每行的最大值和第二大值的索引
    max_values, max_indices = torch.topk(output, k=2, dim=1)

    # 计算最大值和第二大值之间的差异
    diff_values = max_values[:, 0] - max_values[:, 1]

    # 提取每行最大值对应的类别
    predicted_labels = max_indices[:, 0]

    # 检查预测是否正确
    correct_predictions = []
    incorrect_predictions = []

    for i in range(len(label)):
        if predicted_labels[i] == label[i]:
            correct_predictions.append(i)
        else:
            incorrect_predictions.append(i)

    print("正确分类的索引列表:", correct_predictions)
    print("错误分类的索引列表:", incorrect_predictions)

    # 计算正确分类中每行最大值与第二大值的差异
    correct_diff_values = [round(diff_values[i].item(), 2) for i in correct_predictions]
    print("正确分类中每行最大值与第二大值的差异:", correct_diff_values)

    # 计算错误分类中每行最大值与第二大值的差异
    incorrect_diff_values = [round(diff_values[i].item(), 2) for i in incorrect_predictions]
    print("错误分类中每行最大值与第二大值的差异:", incorrect_diff_values)

    # 假设您已经计算了正确分类和错误分类的数量
    total_correct = len(correct_diff_values)
    total_incorrect = len(incorrect_diff_values)

    # 绘制 correct_diff_values 和 incorrect_diff_values 的图像
    plt.figure(figsize=(10, 6))
    plt.plot(correct_diff_values, marker='o', linestyle='-', color='b', label='Correct Diff Values')
    plt.plot(incorrect_diff_values, marker='o', linestyle='-', color='r', label='Incorrect Diff Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Diff Value')
    plt.title(f'Diff Values Comparison({corruption_type})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 计算阈值的上限，即列表中的最大差异
    max_diff = max(max(correct_diff_values), max(incorrect_diff_values))

    # 定义从 0 到最大差异的一系列阈值，步长为 0.1
    thresholds = [i * 0.1 for i in range(int(max_diff * 10) + 1)]

    # 计算每个阈值以上正确分类占总数的比例
    correct_ratios = []
    correct_counts = []  # 用于存储每个阈值以上的正确分类数量
    for threshold in thresholds:
        correct_above_threshold = sum(diff > threshold for diff in correct_diff_values)
        incorrect_above_threshold = sum(diff > threshold for diff in incorrect_diff_values)
        total_above_threshold = correct_above_threshold + incorrect_above_threshold
        correct_ratio = correct_above_threshold / total_above_threshold if total_above_threshold > 0 else 0
        correct_ratios.append(correct_ratio)
        correct_counts.append(correct_above_threshold)  # 存储每个阈值以上的正确分类数量

    # 绘制每个阈值以上的正确分类占总数的比例
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Diff thread hold max1-max2')
    ax1.set_ylabel('correct class ratio', color=color)
    ax1.plot(thresholds, correct_ratios, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 右边的轴用于绘制正确分类数量
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('correct Nums', color=color)
    ax2.plot(thresholds, correct_counts, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Correct Classification Ratio and Count at Different Thresholds({corruption_type})')
    plt.grid(True)  # 只使用纵向网格线
    plt.show()

    print("11111111")



# %%
def generate_weights(N, index, weights, results):
    if index == N:  # 递归终止条件
        if sum(weights) != 0:
        # 12头
        # if sum(weights) != 0 and weights[-1]==1 and weights[-2]==1 and weights[7]==0 and weights[8]==1:  # 排除全0的数组
            results.append(weights[:])
        return

    weights[index] = 0
    generate_weights(N, index + 1, weights, results)

    weights[index] = 1
    generate_weights(N, index + 1, weights, results)

def generate_all_weights(N):
    results = []
    generate_weights(N, 0, [0] * N, results)
    print(results)
    return results


def rankTTA_Acc(filename,filenameOut):
    # 读取文件并解析数据
    data = []
    with open(filename, 'r') as file:
        for line in file:
            match = re.match(r"Weight (.+), err_mean: (\d+\.\d+)%", line.strip())
            if match:
                weights = list(map(int, match.group(1).split(', ')))
                err_mean = float(match.group(2))
                data.append((weights, err_mean))
            else:
                print("Invalid line format:", line.strip())  # 如果数据格式不符合预期，打印错误信息

    # 根据 err_mean 进行排序
    sorted_data = sorted(data, key=lambda x: x[1])

    # 将排序后的结果写入新文件
    with open(filenameOut, 'w') as file:
        for weights, err_mean in sorted_data:
            weights_str = ', '.join(map(str, weights))
            file.write(f"Weight {weights_str}, err_mean: {err_mean:.2f}%\n")

def batchRank():
    filenames = [
        # 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide5/b1-b3ResOut/train30epochs_lr_0.1/TTA_norm_accuracies.txt',
        'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide10/b1-b3ResOut/train30epochs_lr_0.1/TTA_norm_accuracies.txt'
    ]
    filenameOuts = [
        'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide5/b1-b3ResOut/train30epochs_lr_0.1/RankTTA_norm_accuracies.txt',
        'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide10/b1-b3ResOut/train30epochs_lr_0.1/TTA_norm_accuracies.txt'
    ]
    for i in range(len(filenames)):
        rankTTA_Acc(filenames[i], filenameOuts[i])


def compare_files(file1_path, file2_path):
    # 打开并读取两个文件的内容
    with open(file1_path, 'r', encoding='utf-8') as file1:
        file1_content = file1.readlines()

    with open(file2_path, 'r', encoding='utf-8') as file2:
        file2_content = file2.readlines()

    # 比较两个文件的内容
    diff_lines = []
    for line_num, (line1, line2) in enumerate(zip(file1_content, file2_content), start=1):
        if line1 != line2:
            diff_lines.append((line_num, line1, line2))

    return diff_lines

def diff_files():
    # 调用函数比较两个文件的差异
    file1_path = 'initparam.txt'
    file2_path = 'after1epoch.txt'
    file3_path = 'b1训练完.txt'
    file4_path = 'b2训练1epoch.txt'
    file5_path = 'b2训练2epoch.txt'

    differences = compare_files(file3_path, file4_path)

    # 输出差异行
    if differences:
        print("文件存在差异：")
        for line_num, line1, line2 in differences:
            print(f"行 {line_num}:")
            print(f"文件1: {line1.strip()}")
            print(f"文件2: {line2.strip()}")
            print()
    else:
        print("文件相同，没有差异。")

# Name: MINE_simple
# Author: Reacubeth
# Time: 2020/12/15 18:49
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


SIGNAL_NOISE = 0.2
SIGNAL_POWER = 3

data_dim = 3
num_instances = 20000


def gen_x(num, dim):
    return np.random.normal(0., np.sqrt(SIGNAL_POWER), [num, dim])


def gen_y(x, num, dim):
    return x + np.random.normal(0., np.sqrt(SIGNAL_NOISE), [num, dim])


def true_mi(power, noise, dim):
    return dim * 0.5 * np.log2(1 + power/noise)


# mi = true_mi(SIGNAL_POWER, SIGNAL_NOISE, data_dim)
# print('\nTrue MI:', mi)
#
#
# hidden_size = 10
# n_epoch = 500


class MINE(nn.Module):
    def __init__(self, hidden_size=10):
        super(MINE, self).__init__()
        self.layers = nn.Sequential(nn.Linear(2 * data_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x, y, o):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x, ], dim=0).cuda()
        idx = torch.randperm(batch_size)

        tensor_data = torch.tensor(y)

        # 调整张量的形状为 (10000, 1)
        tensor_data = tensor_data.view(len(y), 1)

        y = tensor_data.cuda()

        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        loss = - np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        # compute loss, you'd better scale exp to bit
        return loss

def MINE_info(x,y,o):
    model = MINE(hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    plot_loss = []
    all_mi = []
    for epoch in tqdm(range(n_epoch)):
        # x_sample = torch.from_numpy(x_sample).float()
        # y_sample = torch.from_numpy(y_sample).float()
        x_sample = x
        y_sample = y

        loss = model(x_sample, y_sample, o)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        all_mi.append(-loss.item())


    fig, ax = plt.subplots()
    ax.plot(range(len(all_mi)), all_mi, label='MINE Estimate')
    ax.plot([0, len(all_mi)], [mi, mi], label='True Mutual Information')
    ax.set_xlabel('training steps')
    ax.legend(loc='best')
    plt.show()








if __name__ == '__main__':
    batchRank()
    # diff_files()