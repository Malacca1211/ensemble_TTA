#%%
import pandas as pd

import os
import argparse
import re
import pdb
# get clean error
global sev
# parse data to dict
def parse_data(data):
    parsed_data = {}
    for line in data.split('\n'):
        if 'error %' in line:
            parts = line.split('error %')
            parts = parts[1].split('[')
            noise_type = parts[1].split(']')[0]
            severity = (re.findall(r"\d+",noise_type))
            noise_type = noise_type.split(severity[0])[0]
            if len(severity) > 1:
                severity = severity[0]+severity[1]
            severity=int(severity[0])
            parsed_data.setdefault(noise_type, {})
            error_rate = float(parts[1].split(']: ')[1].split('%')[0])
            parsed_data[noise_type][severity] = error_rate

    return parsed_data


def parse_all_results(record_path):
    configs_=parse_filename(record_path)
    clean_error_=None
    try:
        cln_fp=record_path+'/logs2_tent_Adam.log'
        with open(cln_fp, 'r') as file:
            contents = file.read()

        # 使用正则表达式模块查找匹配的字符串
        match = re.search(r'original best prec1 is ([\d.]+)', contents)

        # 如果找到了匹配的字符串，则输出匹配的数字
        # pdb.set_trace()
        clean_acc_=match.group(1)
        clean_error_=100-float(clean_acc_)
    except:
        pass
        

    # 整理代码
    record_path=record_path+'/TTA'
    files = os.listdir(record_path)
    files.sort()
    fp=[]
    for n in files:
        if n.startswith(f'tent_Adam_{lr}') :#or n.startswith('source_'):
            fp.append(os.path.join(record_path,n))
    
    results = {}
    for n in fp:
        with open(n, 'r') as f:
            data = f.read()
            parsed_data = parse_data(data)
            f.close()   
        tta=n.split('/')[-1].split('_')[0]
        layer_num = configs_['local_module_num']
        results.setdefault(tta, {})
        results[tta][layer_num] = parsed_data
        results[tta][layer_num]['optim']=n.split('/')[-1].split('_')[1]
        results[tta][layer_num]['LR']=float(n.split('/')[-1].split('_')[2])
    # calculate mean error over noise types
    # noise_num_last = 15
    for tta in results.keys():
        for ly in results[tta].keys():
            noise_num = len(results[tta][ly].keys())-2
            # assert noise_num == noise_num_last ,'noise_num not equal'
            
            mean_err = dict.fromkeys([1,5], 0)
            for n in results[tta][ly].keys():
                if (n !=  'optim') and (n != 'LR'):
                    for s in [1,5]:
                        mean_err[s] += results[tta][ly][n][s]
            for s in mean_err.keys():
                mean_err[s] /= noise_num
            results[tta][ly]['mean'] = mean_err
            # noise_num_last = noise_num
    # 建立指定行名和列名的Dataframe
    # inds=[1,2,4,8,16]
    cols=['tta_method','local_modules','aux','ixx1','ixx2','ixy1','ixy2','noise_type','severity','error','clean_error','optim','LR']
    df = pd.DataFrame(columns=cols)
    for tta in results.keys():
        for ly in results[tta].keys():
            for nt in results[tta][ly].keys():
                if (nt !=  'optim') and (nt != 'LR'):
                    for s in results[tta][ly][nt].keys():
                        df = df.append(pd.DataFrame([[tta,ly,configs_['aux_net_config'],configs_['ixx_1'],configs_['ixx_2'],configs_['ixy_1'],configs_['ixy_2'],nt,s,results[tta][ly][nt][s],clean_error_,results[tta][ly]['optim'],results[tta][ly]['LR']]],columns=cols),ignore_index=True)
    return df


def parse_all_results2(record_path):
    clean_error_=None
    try:
        cln_fp=record_path+'/logs2_tent_Adam.log'
        with open(cln_fp, 'r') as file:
            contents = file.read()

        # 使用正则表达式模块查找匹配的字符串
        match = re.search(r'original best prec1 is ([\d.]+)', contents)

        # 如果找到了匹配的字符串，则输出匹配的数字
        # pdb.set_trace()
        clean_acc_=match.group(1)
        clean_error_=100-float(clean_acc_)
    except:
        pass
        
    # 整理代码
    files = os.listdir(record_path)
    files.sort()
    fp=[]
    for n in files:
        if n.startswith(f'tent_Adam_1e-08_bs200_group.txt') :#or n.startswith('source_'):
            fp.append(os.path.join(record_path,n))
    
    results = {}
    for n in fp:
        with open(n, 'r') as f:
            data = f.read()
            parsed_data = parse_data(data)
            f.close()   
        tta=n.split('/')[-1].split('_')[0]
        results.setdefault(tta, {})
        results[tta] = parsed_data
        results[tta]['optim']=n.split('/')[-1].split('_')[1]
        results[tta]['LR']=float(n.split('/')[-1].split('_')[2])
    # calculate mean error over noise types
    # noise_num_last = 15
    for tta in results.keys():
            noise_num = len(results[tta].keys())-2
            # assert noise_num == noise_num_last ,'noise_num not equal'
            
            mean_err = dict.fromkeys([sev], 0)
            for n in results[tta].keys():
                if (n !=  'optim') and (n != 'LR'):
                    for s in [sev]:
                        mean_err[s] += results[tta][n][s]
            for s in mean_err.keys():
                mean_err[s] /= noise_num
            results[tta]['mean'] = mean_err
            # noise_num_last = noise_num
    # 建立指定行名和列名的Dataframe
    # inds=[1,2,4,8,16]
    cols=['noise_type','error']
    df = pd.DataFrame(columns=cols)
    for tta in results.keys():
            for nt in results[tta].keys():
                if (nt !=  'optim') and (nt != 'LR'):
                    for s in [sev]:#results[tta][nt].keys():
                        df = df.append(pd.DataFrame([[nt,results[tta][nt][s]]],columns=cols),ignore_index=True)
    return df


def parse_filename(filename):
    result = {}
    match = re.search(r'K_(\d+)', filename)
    if match:
        result['local_module_num'] = int(match.group(1))
    
    match = re.search(r'aux_net_config_(\w+)', filename)
    if match:
        result['aux_net_config'] = match.group(1)
    
    match = re.search(r'local_loss_mode_(\w+)', filename)
    if match:
        result['local_loss_mode'] = match.group(1)
    
    match = re.search(r'net_widen_([\d.]+)', filename)
    if match:
        result['net_widen'] = float(match.group(1))
    
    match = re.search(r'aux_net_feature_dim_(\d+)', filename)
    if match:
        result['aux_net_feature_dim'] = int(match.group(1))
    
    match = re.search(r'ixx_1_([\d.]+)', filename)
    if match:
        result['ixx_1'] = float(match.group(1))
    
    match = re.search(r'ixy_1_([\d.]+)', filename)
    if match:
        result['ixy_1'] = float(match.group(1))
    
    match = re.search(r'ixx_2_([\d.]+)', filename)
    if match:
        result['ixx_2'] = float(match.group(1))
    
    match = re.search(r'ixy_2_([\d.]+)', filename)
    if match:
        result['ixy_2'] = float(match.group(1))
    
    return result
# if not os.path.exists('results_loctta'):
#     os.makedirs('results_loctta')
# df.to_csv('results_loctta/'+record_path.replace('/','-')+'noauxbn'+'.csv',index=False)

# global lr
# lr='0.0005'

# df=pd.DataFrame()
# record_path1='/home/cll/code/local-learning/InfoPro-Pytorch/Experiments_on_CIFAR-SVHN-STL10/output_/InfoPro_cifar10_wideresnet28_K_2_/no_1_aux_net_config_1c2f_local_loss_mode_cross_entropy_aux_net_widen_1.0_aux_net_feature_dim_128_ixx_1_5.0_ixy_1_0.0_ixx_2_5.0_ixy_2_0.0_cos_lr_'
# df1=parse_all_results(record_path1)
# df=pd.concat([df,df1])
# print(df)

# n='/home/cll/code/local-learning/InfoPro-Pytorch/Experiments_on_CIFAR-SVHN-STL10/tent/output/tent_230506_140812.txt'
# with open(n, 'r') as f:
#     data = f.read()
#     parsed_data = parse_data(data)
#     f.close()   
# df2=pd.DataFrame(parsed_data)
# df2_sev5=df2.loc[5]
# # df2_sev5['mean']=df2_sev5.mean(axis=1)
# df1_sev5=df1.loc[df1['severity']==5][['noise_type','error']]
# df1_sev5.index=df1_sev5['noise_type']
# df1_sev5=df1_sev5['error']
# df2_sev5['mean']=df2_sev5.mean()

# for n in df1.index:
#     df1.loc[n,'mean']=df2_sev5.loc[df1.loc[n,'noise_type']]
sev=1
record_path3='/home/cll/code/course/pytorch-classification/checkpoints/cifar10/resnet-110-group/TTA'
df3=parse_all_results2(record_path3)
df3
# df = pd.merge(df1_sev5, df2_sev5,on=df1_sev5.index)
# df.index=df['key_0']
# df=df.drop(columns=['key_0'])
# df.columns=['local','original']
# # merge num 3
# df_3=pd.merge(df,df3,on=df.index)
# df_3=df_3.drop(columns=['noise_type'])
# df_3.index=df_3['key_0']
# df_3=df_3.drop(columns=['key_0'])
# df_3.columns=['lo cal','original','tune_all']
# # 绘制柱状图
# ax = df_3.plot(kind='bar')
# # 添加标题
# ax.set_title('Error wrt noise type')

# # 倾斜索引标签
# ax.set_xticklabels(df_3.index, rotation=45, ha='right')
# # %%
# fig = ax.get_figure()
# fig.set_size_inches(12, 10)  # 调整图像的宽度和高度
# fig.savefig('Experiments_on_CIFAR-SVHN-STL10/results_t/trial_1_tune_all.png')

# %%
