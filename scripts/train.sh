DEVICES="4"

## cifar10
# CONFIG=configs/cifar10_trip.yaml
CONFIG=configs/cifar10_simclr.yaml
# CONFIG=configs/cifar10_simsiam.yaml

# CONFIG=configs/cifar10_trip_roma.yaml
# CONFIG=configs/cifar10_simclr_roma.yaml
# CONFIG=configs/cifar10_simsiam_roma.yaml

DATA_DIR=/data/ssl/
LOG_DIR=./results_cifar10

## cifar100
# CONFIG=configs/cifar100_trip.yaml
# CONFIG=configs/cifar100_simclr.yaml
# CONFIG=configs/cifar100_simsiam.yaml

# CONFIG=configs/cifar100_trip_roma.yaml
# CONFIG=configs/cifar100_simclr_roma.yaml
# CONFIG=configs/cifar100_simsiam_roma.yaml

# DATA_DIR=/data/ssl/
# LOG_DIR=./results_cifar100


CUDA_VISIBLE_DEVICES=$DEVICES python main.py \
--data_dir $DATA_DIR \
--log_dir $LOG_DIR/logs/ \
-c $CONFIG \
--ckpt_dir $LOG_DIR/ckpts/