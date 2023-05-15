DEVICES="0"
PRETRAIN_PATH=/path/to/pth

# cifar10
CONFIG=configs/cifar10_eval.yaml
LOG_DIR=./results_cifar10
DATA_DIR=/data/ssl/

# cifar100
# CONFIG=configs/cifar100_eval.yaml
# LOG_DIR=./results_cifar100
# DATA_DIR=/data/ssl/

# stl10
# CONFIG=configs/stl10_eval.yaml
# LOG_DIR=./results_stl10
# DATA_DIR=/data/ssl/

# imagenet100
# CONFIG=configs/imagenet100_eval.yaml
# LOG_DIR=./results_imagenet100
# DATA_DIR=/data/ssl/ImageNet2012_100/

# single gpu linear eval
CUDA_VISIBLE_DEVICES=$DEVICES python linear_eval.py \
    --data_dir $DATA_DIR \
    --log_dir $LOG_DIR/test_logs/ \
    -c $CONFIG \
    --ckpt_dir $LOG_DIR/test_ckpt  \
    --eval_from $PRETRAIN_PATH

# DDP linear eval
# CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --master_port 29510 --nproc_per_node 4 linear_eval_ddp.py \
# --data_dir $DATA_DIR \
# --log_dir $LOG_DIR/test_logs/ \
# -c $CONFIG \
# --ckpt_dir $LOG_DIR/test_ckpts// \
# --eval_from $PRETRAIN_PATH