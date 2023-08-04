# <p align=center> :fire: `Domain-irrelevant Feature Learning for Generalizable Pan-sharpening (ACMMMM 2023)`</p>

![Python 3.8](https://img.shields.io/badge/python-3.8-g) ![pytorch 1.12.0](https://img.shields.io/badge/pytorch-1.12.0-blue.svg)

This is the official PyTorch codes for the paper.  
>**Domain-irrelevant Feature Learning for Generalizable Pan-sharpening**<br>  [Yunlong Lin<sup>*</sup>](https://scholar.google.com.hk/citations?user=5F3tICwAAAAJ&hl=zh-CN), [Zhenqi fu<sup>*</sup>](https://zhenqifu.github.io/index.html), [Ge Meng](), [Yingying Wang](), [Yuhang Dong](https://li-chongyi.github.io/), [Linyu Fan](),  [Hedeng Yu](), [Xinghao Ding†](https://scholar.google.com.hk/citations?user=k5hVBfMAAAAJ&hl=zh-CN&oi=ao)（ * co-first author. † indicates corresponding author)<br>

<div align=center><img src="img/overview.jpg" height = "100%" width = "100%"/></div>

### :rocket: Highlights:
- **SOTA performance**: The proposed DIRFL achieves better generalization performance than existing SOTA pan-sharpening methods over multiple satellite datasets.

## Dependencies and Installation
- Ubuntu >= 18.04
- CUDA >= 11.0
```
# git clone this repository
git clone https://github.com/LYL1015/DIRFL.git
cd DIRFL

# create new anaconda env
conda create -n DIRFL python=3.8
conda activate DIRFL

# install python dependencies
pip install -r requirements.txt
```
## Datasets
Training dataset, testing dataset are available at [Data](https://github.com/manman1995/Awaresome-pansharpening).
3. The final directory structure will be arranged as:
```
Data
    |- WV3_data
        |- train128
            |- pan
            |- ms
        |- test128
            |- pan
            |- ms
    |-  WV2_data
        |- train128
            |- pan
            |- ms
        |- test128
            |- pan
            |- ms
    |-  GF2_data
        |- train128
            |- pan
            |- ms
        |- test128
            |- pan
            |- ms
```

## Testing the Model

To test the trained pan-sharpening model, you can run the following command:

```
python test.py
```

## Configuration

The configuration options are stored in the `option.yaml` file and `test.py`. Here is an explanation of each of the options:

#### algorithm

- algorithm: The model for testing

#### Testing

- `algorithm`: The algorithm to use for testing.
- `type`: The type of testing, `test`
- `data_dir`: The location of the test data.
- `source_ms`: The source of the multi-spectral data.
- `source_pan`: The source of the panchromatic data.
- `model`:  The model path to use for testing.
- `save_dir`: The location to save the test results.
- `test_config_path` : The configuration file path for models
  
#### Data Processingc

- `upscale`: The upscale factor.
- `batch_size`: The size of each batch.
- `patch_size`: The size of each patch.
- `data_augmentation`: Whether to use data augmentation.
- `n_colors`: The number of color channels.
- `rgb_range`: The range of the RGB values.
- `normalize`: Whether to normalize the data.
