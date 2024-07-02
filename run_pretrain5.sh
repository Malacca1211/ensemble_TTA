
#python cifar5.py --lr 0.1 --epoch 250  --manualSeed 13  --arch 'resnet56_SDN' --add_ic 'b1b2b3b4b5b6'
#python cifar5.py --lr 0.1 --epoch 250  --manualSeed 14  --arch 'resnet56_SDN' --add_ic 'b1b2b3b4b5b6'


python findBestAcc.py --lr 0.001 --step 60  --gamma 0.1 --epochs 120
python findBestAcc.py --lr 0.001 --step 60  --gamma 0.2 --epochs 120
python findBestAcc.py --lr 0.0005 --step 60  --gamma 0.3 --epochs 120


#python findBestAcc.py --lr 0.001 --step 60  --gamma 0.25
#
#python findBestAcc.py --lr 0.001 --step 40  --gamma 0.1