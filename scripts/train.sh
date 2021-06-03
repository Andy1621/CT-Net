###
 # @Author: Kunchang Li
 # @Date: 2021-04-15 22:54:18
 # @LastEditors: Kunchang Li
 # @LastEditTime: 2021-04-16 20:39:59
### 

# something
########################################################DP########################################################
# warmup 10epoch+cos decay 30epoch
# parallel x 1x3x3&3x1x1 + y 1x3x3&3x1x1 + 1x1x1 fusion + spatiotemporal group attention plus + replace 7
# 8帧
# cos_decay
# lr: 0~9 0~0.02 10-44 0.02~0
# dropout: 0.3
# wd: 5e-4
# top1: 0%
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 main.py something RGB \
     --root-log ./log \
     --root-model ./model \
     --arch resnet50 --model CT_Net --num-segments 8 \
     --gd 20 --lr 0.02 --unfrozen-epoch 0 --lr-type cos \
     --warmup 10 --tune-epoch 10 --tune-lr 0.02 --epochs 45 \
     --batch-size 8 -j 24 --dropout 0.3 --consensus-type=avg \
     --npb --num-total 7 --full-res --gpus 0 1 2 3 4 5 6 7 --suffix 2021
     
########################################################DDP########################################################
# warmup 10epoch+cos decay 30epoch
# parallel x 1x3x3&3x1x1 + y 1x3x3&3x1x1 + 1x1x1 fusion + spatiotemporal group attention plus + replace 7
# 8帧
# cos_decay
# lr: 0~9 0~0.02 10-44 0.02~0
# dropout: 0.3
# wd: 5e-4
# top1: 0%
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=8 \
    main_ddp.py something RGB \
     --root-log .log \
     --root-model ./model \
     --arch resnet50 --model CT_Net --num-segments 8 \
     --gd 20 --lr 0.02 --unfrozen-epoch 0 --lr-type cos \
     --warmup 10 --tune-epoch 10 --tune-lr 0.02 --epochs 45 \
     --batch-size 8 -j 24 --dropout 0.3 --consensus-type=avg \
     --npb --num-total 7 --full-res --gpus 0 1 2 3 4 5 6 7 --suffix 2021



# kinetics
########################################################DDP########################################################
# warmup 10epoch+cos decay 80epoch
# parallel x 1x3x3&3x1x1 + y 1x3x3&3x1x1 + 1x1x1 fusion + spatiotemporal group attention + replace 7
# 8帧
# cos_decay
# lr: 10~89 0.01~0
# dropout: 0.5
# wd: 1e-4
# top1: 0%
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=8 \
     --root-log .log \
     --root-model ./model \
     --arch resnet50 --model CT_Net --num-segments 8 \
     --gd 20 --lr 0.01 --unfrozen-epoch 0 --wd 1e-4 --lr-type cos \
     --warmup 10 --tune-epoch 90 --tune-lr 0 --epochs 90 \
     --batch-size 64 -j 16 --dropout 0.5 --consensus-type=avg \
     --npb --num-total 7 --full-res --dense-sample --gpus 0 1 2 3 4 5 6 7 --suffix 2021