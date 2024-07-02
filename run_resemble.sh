
CUDA_VISIBLE_DEVICES=1 python cifar10c_all_resemble.py -a resnet --depth 110 --model-paths \
 checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_1/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_6/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_48/model_best.pth.tar \
   checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_90/model_best.pth.tar \
 --out-dir results_resemble/cifar10/diff_seed --cfg tent/cfgs/norm.yaml OPTIM.LR 0.0005 TEST.BATCH_SIZE 200 

CUDA_VISIBLE_DEVICES=3 python cifar10c_all_resemble.py -a resnet --depth 110 --model-paths \
 checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_1/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.2_wd0.0001_CosALR_120/seed_49/model_best.pth.tar \
   checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.2_wd0.0001_CosALR_164/seed_2/model_best.pth.tar \
   checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.3_wd0.0001_CosALR_164/seed_2/model_best.pth.tar \
 --out-dir results_resemble/cifar10/mean_softmax/diff_params --cfg tent/cfgs/norm.yaml OPTIM.LR 0.0005 TEST.BATCH_SIZE 200 

CUDA_VISIBLE_DEVICES=3 python cifar10c_all_resemble.py -a resnet --depth 110 --model-paths \
 checkpoint/cifar10/resnet32/cifar10/resnet_w4/SGD/bs128_lr0.1_wd0.0001_CosALR_164/seed_6/model_best.pth.tar \
 checkpoint/cifar10/resnet56/cifar10/resnet_w4/SGD/bs128_lr0.1_wd0.0001_CosALR_164/seed_0/model_best.pth.tar \
 checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_1/model_best.pth.tar \
 --out-dir results_resemble/cifar10/mean_softmax/diff_arch --cfg tent/cfgs/norm.yaml OPTIM.LR 0.0005 TEST.BATCH_SIZE 200 
#  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_1/model_best.pth.tar \


CUDA_VISIBLE_DEVICES=1 python cifar10c_all.py -a resnet --depth 110 --model-path checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_1 --cfg tent/cfgs/tent.yaml OPTIM.LR 0.0001 TEST.BATCH_SIZE 200 


python cifar.py -a resnet --depth 110 --epochs 164 --lr-scheduler CosALR --gamma 0.1 --wd 1e-4 --checkpoint checkpoint/cifar10/resnet110-bottleneck --gpu-id 3 --block-name Bottleneck
python cifar.py -a resnet --depth 32 --epochs 164 --lr-scheduler CosALR --gamma 0.1 --wd 1e-4 --checkpoint checkpoint/cifar10/resnet32 --gpu-id 1 --manualSeed 6
python cifar.py -a resnet --depth 56 --epochs 164 --lr-scheduler CosALR --gamma 0.1 --wd 1e-4 --checkpoint checkpoint/cifar10/resnet56-bottleneck --gpu-id 1 --manualSeed 32 --block-name Bottleneck

CUDA_VISIBLE_DEVICES=1 python cifar10c_all_resemble.py -a resnet --depth 110 --model-paths checkpoint/cifar10/resnet110/resnet_w1/SGD/bs64_lr0.1_wd0.0001_StepLR_60_0.1/seed_4/model_best.pth.tar checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_1/model_best.pth.tar checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_CosALR_164/seed_2/model_best.pth.tar checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_CosALR_100/seed_8/model_best.pth.tar  --cfg tent/cfgs/norm.yaml OPTIM.LR 0.0005 TEST.BATCH_SIZE 200 

CUDA_VISIBLE_DEVICES=3 python cifar10c_all_resemble.py -a resnet --depth 110 --model-paths \
 checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_1/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_6/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_48/model_best.pth.tar \
   checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_90/model_best.pth.tar \
   --ensemble-mode majority_vote --out-dir results_resemble/cifar10/majority_vote/diff_seed --cfg tent/cfgs/norm.yaml OPTIM.LR 0.0005 TEST.BATCH_SIZE 200 

CUDA_VISIBLE_DEVICES=3 python cifar10c_all_resemble.py -a resnet --depth 110 --model-paths \
 checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_1/model_best.pth.tar \
   --ensemble-mode majority_vote --out-dir results_resemble/cifar10/majority_vote/diff_seed --cfg tent/cfgs/norm.yaml OPTIM.LR 0.0005 TEST.BATCH_SIZE 200 

CUDA_VISIBLE_DEVICES=3 python cifar10c_all_resemble.py -a resnet --depth 110 --model-paths \
 checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_1/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_6/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_48/model_best.pth.tar \
   checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_90/model_best.pth.tar \
   --ensemble-mode mean_softmax --out-dir results_resemble/cifar10/mean_softmax/diff_seed --cfg tent/cfgs/tent.yaml OPTIM.LR 0.0005 TEST.BATCH_SIZE 200 

CUDA_VISIBLE_DEVICES=3 python cifar10c_all_resemble.py -a resnet --depth 110 --model-paths \
 checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_1/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_6/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_48/model_best.pth.tar \
   checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_90/model_best.pth.tar \
   --ensemble-mode mean_softmax --out-dir results_resemble/cifar10/mean_softmax/diff_seed --cfg tent/cfgs/tent.yaml OPTIM.LR 0.0005 TEST.BATCH_SIZE 200 

CUDA_VISIBLE_DEVICES=1 python cifar10c_all_resemble.py -a resnet --depth 110 --model-paths \
 checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_1/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_6/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_48/model_best.pth.tar \
   checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_90/model_best.pth.tar \
   --ensemble-mode mean_softmax --out-dir results_resemble/cifar10/mean_softmax/diff_seed --cfg tent/cfgs/tent.yaml OPTIM.LR 0.0005 TEST.BATCH_SIZE 200 MODEL.ADAPTATION tent_mean_out

   CUDA_VISIBLE_DEVICES=2 python cifar10c_all_resemble.py -a resnet --depth 110 --model-paths \
 checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_1/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_6/model_best.pth.tar \
  checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_48/model_best.pth.tar \
   checkpoint/cifar10/resnet110/resnet_w1/SGD/bs128_lr0.1_wd0.0001_StepLR_60_0.1/seed_90/model_best.pth.tar \
 --out-dir results_resemble/cifar10/diff_seed/source_data --eval-source --cfg tent/cfgs/norm.yaml OPTIM.LR 0.0005 TEST.BATCH_SIZE 200 