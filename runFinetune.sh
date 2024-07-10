#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/addIc_b2b3b4b5b6'
#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/addIc_b3b4b5b6'
#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/addIc_b3b5'
#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/addIc_b5b6'

#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/addIc_b1b2b3b4b5b6'
#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/addIc_b2b4b6'


#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneNewout/addIc_b3b4b5b6NewOut'
#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneNewout/addIc_b4b5b6NewOut'

#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_14/finetuneResout10/addIc_b2b3b4ResOut' --interOuts 10
#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneNewout16/addIc_b2b3b4NewOut' --interOuts 16


#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout8/addIc_b3b4b5b6ResOut' --interOuts 8  --epochs 60 --lr 0.01 --step 120 --gamma 0.1
#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout8/addIc_b3b4b5b6ResOut' --interOuts 8  --epochs 80 --lr 0.01 --step 60 --gamma 0.25
#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout8/addIc_b3b4b5b6ResOut' --interOuts 8  --epochs 80 --lr 0.01 --step 60 --gamma 0.1
#python finetune.py  --model_path 'checkpoint/cifar10/resnet56_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout8/addIc_b3b4b5b6ResOut' --interOuts 8  --epochs 60 --lr 0.005 --step 120 --gamma 0.1


#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout0/addIc_b1b2ResOut' --interOuts 0 --epochs 50 --lr 0.1

#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout2/addIc_b1b2ResOut' --interOuts 2 --epochs 100 --lr 0.01 --step 40 --gamma 0.2

#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout4/addIc_b1b2ResOut' --interOuts 4 --epochs 100 --lr 0.01 --step 40 --gamma 0.2
#
#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout6/addIc_b1b2ResOut' --interOuts 6 --epochs 100 --lr 0.01 --step 40 --gamma 0.2

#python cifar10c_all_resemble.py  --model-path 'finetune/cifar10/WideResnet26/finetuneCheckpoint/b1b2ResOut0/train60epochs_lr_0.1/outputIndex_2_bestModel' --interOuts 0

#python cifar10c_all_resemble.py  --model-path 'finetune/cifar10/WideResnet26/finetuneCheckpoint/b1b2ResOut2/train100epochs_lr_0.01/outputIndex_2_bestModel' --interOuts 2
#
#python cifar10c_all_resemble.py  --model-path 'finetune/cifar10/WideResnet26/finetuneCheckpoint/b1b2ResOut4/train100epochs_lr_0.01/outputIndex_2_bestModel' --interOuts 4
#
#python cifar10c_all_resemble.py  --model-path 'finetune/cifar10/WideResnet26/finetuneCheckpoint/b1b2ResOut6/train100epochs_lr_0.01/outputIndex_2_bestModel' --interOuts 6

#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout2/addIc_b1b2ResOut' --interOuts 2 --epochs 3 --lr 0.1

#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout0/Wide2/addIc_b1b2ResOut' --interOuts 0 --epochs 30 --lr 0.1  --seed 13
#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_14/finetuneResout0/Wide2/addIc_b1b2ResOut' --interOuts 0 --epochs 30 --lr 0.1  --seed 14
#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_15/finetuneResout0/Wide2/addIc_b1b2ResOut' --interOuts 0 --epochs 30 --lr 0.1  --seed 15
#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_16/finetuneResout0/Wide2/addIc_b1b2ResOut' --interOuts 0 --epochs 30 --lr 0.1  --seed 16
#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_17/finetuneResout0/Wide2/addIc_b1b2ResOut' --interOuts 0 --epochs 30 --lr 0.1  --seed 17

#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_13/finetuneResout0/wide10/addIc_b1-b11ResOut' --interOuts 0 --epochs 30 --lr 0.1  --seed 13
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w2/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_22/finetuneResout0/Wide2/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 2 --epochs 30 --lr 0.2  --seed 22
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w2/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_23/finetuneResout0/Wide2/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 2 --epochs 30 --lr 0.2  --seed 23

#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w5/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_20/finetuneResout0/Wide5/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 5 --epochs 30 --lr 0.1  --seed 20
#

#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w8/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_20/finetuneResout0/Wide8/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 8 --epochs 20 --lr 0.2  --seed 20
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w9/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_20/finetuneResout0/Wide9/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 9 --epochs 20 --lr 0.2  --seed 20
#python finetuneWideResTinyImage.py --gpu-id 0 --model_path '/home/xietong/ensemble_TTA/checkpoint/tinyImagenet/wideResnet_SDN_w10/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_26/finetuneResout0/Wide10/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 10 --epochs 30 --lr 0.2  --seed 20
#python finetuneWideResTinyImage.py --gpu-id 1 --model_path '/home/xietong/ensemble_TTA/checkpoint/tinyImagenet/wideResnet_SDN_w10/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_26/finetuneResout0/Wide10/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 10 --epochs 20 --lr 0.1  --seed 20
#python finetuneWideResTinyImage.py --gpu-id 2 --model_path '/home/xietong/ensemble_TTA/checkpoint/tinyImagenet/wideResnet_SDN_w10/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_26/finetuneResout0/Wide10/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 10 --epochs 30 --lr 0.1  --seed 20
python finetuneWideResTinyImage.py --gpu-id 3 --model_path '/home/xietong/ensemble_TTA/checkpoint/tinyImagenet/wideResnet_SDN_w10/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_26/finetuneResout0/Wide10/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 10 --epochs 40 --lr 0.2  --seed 20
#python finetuneWideResTinyImage.py  --model_path '/home/xietong/ensemble_TTA/checkpoint/tinyImagenet/wideResnet_SDN_w10/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_26/finetuneResout0/Wide10/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 10 --epochs 20 --lr 0.2  --seed 20
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w10/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_21/finetuneResout0/Wide10/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 10 --epochs 20 --lr 0.1  --seed 21


#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w3/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_20/finetuneResout0/Wide3/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 3 --epochs 30 --lr 0.2  --seed 20
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w3/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_21/finetuneResout0/Wide3/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 3 --epochs 20 --lr 0.2  --seed 21
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w3/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_22/finetuneResout0/Wide3/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 3 --epochs 20 --lr 0.2  --seed 22
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w3/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_23/finetuneResout0/Wide3/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 3 --epochs 20 --lr 0.2  --seed 23
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w3/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_24/finetuneResout0/Wide3/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 3 --epochs 20 --lr 0.2  --seed 24
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w3/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_25/finetuneResout0/Wide3/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 3 --epochs 20 --lr 0.2  --seed 25
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w3/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_26/finetuneResout0/Wide3/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 3 --epochs 20 --lr 0.2  --seed 26


#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w4/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_20/finetuneResout0/Wide4/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 4 --epochs 20 --lr 0.2  --seed 20
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w4/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_21/finetuneResout0/Wide4/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 4 --epochs 20 --lr 0.2  --seed 21
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w4/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_22/finetuneResout0/Wide4/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 4 --epochs 20 --lr 0.2  --seed 22
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w4/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_23/finetuneResout0/Wide4/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 4 --epochs 20 --lr 0.2  --seed 23
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w4/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_24/finetuneResout0/Wide4/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 4 --epochs 20 --lr 0.2  --seed 24
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w4/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_25/finetuneResout0/Wide4/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 4 --epochs 20 --lr 0.2  --seed 25



#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w2/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_23/finetuneResout0/Wide2/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 2 --epochs 20 --lr 0.2  --seed 23
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w2/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_24/finetuneResout0/Wide2/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 2 --epochs 20 --lr 0.2  --seed 24
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w2/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_25/finetuneResout0/Wide2/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 2 --epochs 20 --lr 0.2  --seed 25
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w2/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_26/finetuneResout0/Wide2/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 2 --epochs 20 --lr 0.2  --seed 26
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w2/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_27/finetuneResout0/Wide2/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 2 --epochs 20 --lr 0.2  --seed 27
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w2/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_28/finetuneResout0/Wide2/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 2 --epochs 20 --lr 0.2  --seed 28
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w2/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_29/finetuneResout0/Wide2/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 2 --epochs 20 --lr 0.2  --seed 29

#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w5/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_20/finetuneResout0/Wide5/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 5 --epochs 30 --lr 0.2  --seed 20
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w5/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_21/finetuneResout0/Wide5/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 5 --epochs 30 --lr 0.2  --seed 21
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w5/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_22/finetuneResout0/Wide5/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 5 --epochs 30 --lr 0.2  --seed 22
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w5/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_23/finetuneResout0/Wide5/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 5 --epochs 30 --lr 0.2  --seed 23
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w5/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_24/finetuneResout0/Wide5/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 5 --epochs 30 --lr 0.2  --seed 24


#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w6/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_20/finetuneResout0/Wide6/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 6 --epochs 30 --lr 0.2  --seed 20
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w6/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_21/finetuneResout0/Wide6/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 6 --epochs 30 --lr 0.2  --seed 21
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w6/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_22/finetuneResout0/Wide6/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 6 --epochs 30 --lr 0.2  --seed 22

#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w7/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_20/finetuneResout0/Wide7/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 7 --epochs 30 --lr 0.2  --seed 20
#python finetuneWideResCifar100.py  --model_path 'checkpoint/cifar100/wideResnet_SDN_w7/SGD/bs128_lr0.2_wd0.0005StepLR_60_0.1/seed_21/finetuneResout0/Wide7/addIc_b1-b3ResOut' --interOuts 0 --widen_factor 7 --epochs 30 --lr 0.2  --seed 21


#python finetuneWideRes.py  --model_path 'checkpoint/cifar10/wideResnet_SDN_w4/SGD/bs128_lr0.1_wd0.0005StepLR_60_0.1/seed_14/finetuneResout0/wide2/addIc_b1b2b3ResOut' --interOuts 0 --epochs 30 --lr 0.1  --seed 14
