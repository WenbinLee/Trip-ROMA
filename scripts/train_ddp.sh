DEVICES="0,3"
GPUS=2
PORT=29515

## imagenet
# CONFIG=configs/imagenet100_trip.yaml
CONFIG=configs/imagenet100_simclr.yaml
# CONFIG=configs/imagenet100_simsiam.yaml

# CONFIG=configs/imagenet100_trip_roma.yaml
# CONFIG=configs/imagenet100_simclr_roma.yaml
# CONFIG=configs/imagenet100_simsiam_roma.yaml

DATA_DIR=/data/ssl/ImageNet2012_100/
LOG_DIR=./results_imagenet100

## stl10
# CONFIG=configs/stl10_trip.yaml
# CONFIG=configs/stl10_simclr.yaml
# CONFIG=configs/stl10_simsiam.yaml

# CONFIG=configs/stl10_trip_roma.yaml
# CONFIG=configs/stl10_simclr_roma.yaml
# CONFIG=configs/stl10_simsiam_roma.yaml

# DATA_DIR=/data/ssl/
# LOG_DIR=./results_stl10


CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch --master_port $PORT --nproc_per_node $GPUS main_ddp.py \
--data_dir $DATA_DIR \
--log_dir $LOG_DIR/logs/ \
-c $CONFIG \
--ckpt_dir $LOG_DIR/ckpts/