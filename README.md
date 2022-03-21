# pytorch-WeatherNet ![alt text](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

Inofficial PyTorch implementation of [CNN-based Lidar Point Cloud De-Noising in Adverse Weather](https://arxiv.org/pdf/1912.03874.pdf) (Heinzler et al., 2019). [pytorch-LiLaNet](https://github.com/TheCodez/pytorch-LiLaNet)'s repo is used as base code for this repo and necessary modifications are performed following the instructions in the original paper.

## Differences:

The Autolabeling process is currently not used. For better convergence we add batch normalization after each convolutional layer.

## Dataset

Information: Click [here](https://www.uni-ulm.de/index.php?id=101568) for registration and download.

## Results:

|              | Clear      | Rainy | Foggy  |
|:------------:|----------|------------|----------|
| WeatherNet      |   88.1   |  83.1  |   70.5   |

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the [DENSE CNN denoising](https://www.uni-ulm.de/index.php?id=101568) dataset

## Usage

Train model: train_dense.py trains the model on the given dataset/directory

**Important**: The ```dataset-dir``` must contain the ```train_01```, ```test_01``` and the ```val_01``` folder.

```bash
python train_dense.py
```

Test model: test.py opens a saved model, classifies points in an HDF5 file, and saves the resulting classified points.

ROS Pipeline: ros_test.py runs a ros pipeline subscribing to a PointCloud2 topic and classifies/denoises the points beforepublishing a new PointCloud2 topic.
