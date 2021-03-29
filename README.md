# SpCode-CNN
The python code of  "Sparse Coding-Based Convolutional Neural Network for Image Super-Resolution"

### Created on Thu Mar 25 20:14:42 2021

### @author: 西电博巍(Bowei Wang, QQ: 月光下的云海)

### Version: Ultimate

## usage
Tensorflow-GPU == 1.14.0 and python == 3.8.5

## Download Weight File
The weight file has been uploaded to https://pan.baidu.com/s/1faKxSpFlRQKqqU0fUWf8g (igk4 )

## Instructions for use
### 1. Use pre-trained weights
Download weight file, put in TRAINED_MODEL, run test.py, enter
```
python test.py --which_model SpCode-VDSR --up_scale 4 --file_name ./DATABASE/Set5/woman_GT.bmp 
```

### 2. Use self-trained weights
If you need to configure parameters, modify config.json

```
{
    "blk_size": 16,
    "downscale_fn": "DownScale",
    "downscale_fn_option":["DownScale","DownScale_n_GaussianBlur","GaussianBlur","AddGaussNoise","DownScale_n_GaussianBlur_n_AddGaussNoise"],
    "gauss_radius": 1.5,
	"data_dir": "./DATABASE/DIV2K100/",
    "up_scale": 4,
    "srcnn_config":
        {
            "name": "SRCNN",
            "blk_size":32,
            "test_ratio":0.2,
            "dataset": "./DATABASE/DIV2K100/",
            "epoch": 50000,
            "iter_view":500,
            "lr": 0.001,
            "batch_size": 128,
            "remark": "No remarks yet."
        },
    "vdsr_config":
        {
            "name": "VDSR",
            "blk_size":32,
            "test_ratio":0.2,
            "dataset": "./DATABASE/DIV2K100/",
            "epoch": 5000,
            "iter_view":500,
            "lr": 0.001,
            "batch_size": 64,
            "remark": "No remarks yet."
        },
    "spcode_srcnn":
        {
            "name": "SpCode-SRCNN",
            "test_ratio":0.2,
            "epoch": 50000,
            "iter_view":500,
            "lr": 0.001,
            "batch_size": 64,
            "remark": "No remarks yet."
        },
    "spcode_vdsr":
        {
            "name": "SpCode-VDSR",
            "test_ratio":0.2,
            "epoch": 50000,
            "iter_view":500,
            "lr": 0.001,
            "batch_size": 64,
            "remark": "No remarks yet."
        }
}
```
First, preprocess image. Enter

```python preprocess.py -NI 10 -TD ./DATABASE/ ```

Then run train.py, enter
```
python train.py --which_model SpCode-VDSR
```
to start training.
