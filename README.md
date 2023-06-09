
## Trip-ROMA: Self-Supervised Learning with Triplets and Random Mappings
This is a PyTorch implementation of the [Trip-ROMA](https://openreview.net/forum?id=MR4glug5GU) paper.

<img src='flowchart.png' width=600/>

### Dependencies

If you don't have python 3 environment:
```
conda create -n trip python=3.8
conda activate trip
```
Then install the required packages:
```
pip install -r requirements.txt
```

### Train

1. Config `CONFIG`, `DATA_DIR` and `LOG_DIR` in `script/train.sh` or `script/train_ddp.sh`. We use DDP for ImageNet and STL10 by default.
2. Run
    ```shell
    sh scripts/train.sh
    ```
    for single gpu training, or
    ```shell
    sh scripts/train_ddp.sh
    ```
    for DDP training.

### Test
1. Config `CONFIG`, `DATA_DIR` and `LOG_DIR` in `script/eval.sh`.
2. Run
    ```shell
    sh scripts/eval.sh
    ```

### Citation

```
@article{
li2023triproma,
title={Trip-{ROMA}: Self-Supervised Learning with Triplets and Random Mappings},
author={Wenbin Li and Xuesong Yang and Meihao Kong and Lei Wang and Jing Huo and Yang Gao and Jiebo Luo},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023}
}
```

### Acknowledgement
[PatrickHua/SimSiam](https://github.com/PatrickHua/SimSiam)
