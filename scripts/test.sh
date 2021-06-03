###
 # @Author: Kunchang Li
 # @Date: 2021-04-16 19:35:38
 # @LastEditors: Kunchang Li
 # @LastEditTime: 2021-04-16 20:05:48

########################################################################################################
# CT_Net + replace 7 + 8
# 8x1 256x256
# 49.94% 77.83%
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 test_acc.py something RGB \
     --arch resnet50 --model CT_Net --num-segments 8 \
     --batch-size 64 -j 8 --consensus-type=avg \
     --resume ./model/ct_net_8f_r50.pth.tar\
     --npb --num-total 7 --evaluate --test-crops 1 --full-res --gpus 0 1 2 3 4 5 6 7

# CT_Net + replace 7 + 8
# 8x1 224x224
# 48.39% 76.65%
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 test_acc.py something RGB \
     --arch resnet50 --model CT_Net --num-segments 8 \
     --batch-size 64 -j 8 --consensus-type=avg \
     --resume ./model/ct_net_8f_r50.pth.tar \
     --npb --num-total 7 --evaluate --test-crops 1 --gpus 0 1 2 3 4 5 6 7

# CT_Net + replace 7 + 8
# 8x1x2 256x256
# 50.84% 78.19%
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 test_acc.py something RGB \
     --arch resnet50 --model CT_Net --num-segments 8 \
     --batch-size 64 -j 8 --consensus-type=avg \
     --resume ./model/ct_net_8f_r50.pth.tar \
     --npb --num-total 7 --evaluate --twice-sample --test-crops 1 --full-res --gpus 0 1 2 3 4 5 6 7

# CT_Net + replace 7 + 8
# 8x1x4 256x256
# 51.10% 78.93%      
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 test_acc.py something RGB \
     --arch resnet50 --model CT_Net --num-segments 8 \
     --batch-size 16 -j 8 --consensus-type=avg \
     --resume ./model/ct_net_8f_r50.pth.tar \
     --npb --num-total 7 --evaluate --dense-sample --test-crops 1 --full-res --gpus 0 1 2 3 4 5 6 7

# CT_Net + replace 7 + 8
# 8x3x2 256x256
# 50.97% 79.43%
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 test_acc.py something RGB \
     --arch resnet50 --model CT_Net --num-segments 8 \
     --batch-size 16 -j 8 --consensus-type=avg \
     --resume ./model/ct_net_8f_r50.pth.tar \
     --npb --num-total 7 --evaluate --twice-sample --test-crops 3 --full-res --gpus 0 1 2 3 4 5 6 7

# CT_Net + replace 7 + 8
# 8x3x4 256x256
# 51.23% 79.25%      
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 test_acc.py something RGB \
     --arch resnet50 --model CT_Net --num-segments 8 \
     --batch-size 16 -j 8 --consensus-type=avg \
     --resume ./model/ct_net_8f_r50.pth.tar \
     --npb --num-total 7 --evaluate --dense-sample --test-crops 3 --full-res --gpus 0 1 2 3 4 5 6 7