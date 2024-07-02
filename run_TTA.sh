#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout0/addIc_b1b2ResOut' --interOuts 0 --epochs 30 --lr 0.1 --step 40 --gamma 0.2
#
#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout0/addIc_b1b2ResOut' --interOuts 0 --epochs 60 --lr 0.1 --step 40 --gamma 0.1
#
#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout2/addIc_b1b2ResOut' --interOuts 2 --epochs 30 --lr 0.01 --step 40 --gamma 0.2

#python cifar10c_all_resemble.py  --model-path 'finetune/cifar10/WideResnet26/finetuneCheckpoint/seed13/wide2/b1-b3ResOut/train30epochs_lr_0.1/outputIndex_3_bestModel' --interOuts 0

#python cifar100c_all_resemble.py  --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide2/b1-b3ResOut/train30epochs_lr_0.1/outputIndex_3_bestModel' --interOuts 0 --widen_factor 2
#
#python cifar100c_all_resemble.py  --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide5/b1-b3ResOut/train30epochs_lr_0.1/outputIndex_3_bestModel' --interOuts 0 --widen_factor 5
#
#python cifar100c_all_resemble.py  --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide10/b1-b3ResOut/train30epochs_lr_0.1/outputIndex_3_bestModel' --interOuts 0 --widen_factor 10

#python cifar100c_all_resemble.py --widen_factor 3

python tinyImagenetC_all_resemble.py  --model-path 'finetune/tinyImagenet/WideResnet26/finetuneCheckpoint/seed20/wide10/b1-b3ResOut/train20epochs_lr_0.2/outputIndex_3_bestModel' --interOuts 0 --widen_factor 10


# 初始化学习率
#lr="0.00015" #8e-5
#
## 定义比较函数
#function compare {
#    result=$(awk -v n1="$1" -v n2="0.001" 'BEGIN {if (n1 <= n2) print "true"; else print "false"}') #5e-3
#    if [ "$result" = "true" ]; then
#        return 0
#    else
#        return 1
#    fi
#}
#
# 循环条件：学习率小于等于2e-2
#while compare "$lr"; do
#    echo "Learning rate: $lr"
##    # 更新学习率，每次乘以2
##    python cifar100c_all_resemble.py --tentLr $lr  --weights [0,0,0,1] --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide5/b1-b3ResOut/train30epochs_lr_0.2/outputIndex_3_bestModel' --interOuts 0 --widen_factor 5
##    python cifar100c_all_resemble.py --tentLr $lr  --weights [0.2,0,1,1] --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide5/b1-b3ResOut/train30epochs_lr_0.2/outputIndex_3_bestModel' --interOuts 0 --widen_factor 5
##
##    python cifar100c_all_resemble.py --tentLr $lr  --weights [0,0,0,1] --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide6/b1-b3ResOut/train30epochs_lr_0.2/outputIndex_3_bestModel' --interOuts 0 --widen_factor 6
##    python cifar100c_all_resemble.py --tentLr $lr  --weights [0.2,0,1,1] --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide6/b1-b3ResOut/train30epochs_lr_0.2/outputIndex_3_bestModel' --interOuts 0 --widen_factor 6
##
##    python cifar100c_all_resemble.py --tentLr $lr  --weights [0,0,0,1] --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide7/b1-b3ResOut/train30epochs_lr_0.2/outputIndex_3_bestModel' --interOuts 0 --widen_factor 7
##    python cifar100c_all_resemble.py --tentLr $lr  --weights [0.2,0,1,1] --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide7/b1-b3ResOut/train30epochs_lr_0.2/outputIndex_3_bestModel' --interOuts 0 --widen_factor 7
##
##    python cifar100c_all_resemble.py --tentLr $lr  --weights [0,0,0,1] --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide4/b1-b3ResOut/train20epochs_lr_0.2/outputIndex_3_bestModel' --interOuts 0 --widen_factor 4
##    python cifar100c_all_resemble.py --tentLr $lr  --weights [0.2,0,1,1] --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide4/b1-b3ResOut/train20epochs_lr_0.2/outputIndex_3_bestModel' --interOuts 0 --widen_factor 4
#
#    python cifar100c_all_resemble.py --tentLr $lr  --weights [0,0,0,1] --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide10/b1-b3ResOut/train30epochs_lr_0.1/outputIndex_3_bestModel' --interOuts 0 --widen_factor 10
#    python cifar100c_all_resemble.py --tentLr $lr  --weights [0.2,0,1,1] --model-path 'finetune/cifar100/WideResnet26/finetuneCheckpoint/seed20/wide10/b1-b3ResOut/train30epochs_lr_0.1/outputIndex_3_bestModel' --interOuts 0 --widen_factor 10
#
##    python cifar10c_all_resemble.py --tentLr $lr  --weights [1,1,1,1,1]
#    lr=$(awk -v n="$lr" 'BEGIN {print n + 1e-5}') #+ 1e-4
#done

# 设置外层循环的起始和结束值
#outer_start=0.3
#outer_end=0.5
#
# 设置外层循环的步长
#outer_step=0.1
#
# 设置内层循环的起始和结束值
#inner_start=0.4
#inner_end=0.6
#
# 设置内层循环的步长
#inner_step=0.1
#
# 外层循环
#outer=$outer_start
#while (( $(awk -v outer="$outer" -v outer_end="$outer_end" 'BEGIN {print (outer <= outer_end)}') ))
#do
#    echo "外层循环: $outer"
#
#    # 内层循环
#    inner=$inner_start
#    while (( $(awk -v inner="$inner" -v inner_end="$inner_end" 'BEGIN {print (inner <= inner_end)}') ))
#    do
#        echo "  内层循环: $inner"
#        python cifar10c_all_resemble.py --tentLr 0.0002  --weights [$outer,$inner,0.6,1]
#        python cifar10c_all_resemble.py --tentLr 0.0002  --weights [$outer,$inner,0.7,1]
#        python cifar10c_all_resemble.py --tentLr 0.0002  --weights [$outer,$inner,0.8,1]
#        inner=$(awk -v inner="$inner" -v inner_step="$inner_step" 'BEGIN {print (inner + inner_step)}')
#    done
#
#    outer=$(awk -v outer="$outer" -v outer_step="$outer_step" 'BEGIN {print (outer + outer_step)}')
#done