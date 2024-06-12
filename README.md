# MRGGNet
![Example 1](display/car.gif)
![Example 2](display/chair.gif)
![Example 3](display/plane.gif)
![Example 4](display/table.gif)

## Citing this work
If you find our code or paper useful, please consider citing
```
@article{li2024multi,
  title={Multi-granularity relationship reasoning network for high-fidelity 3D shape reconstruction},
  author={Li, Lei and Zhou, Zhiyuan and Wu, Suping and Li, Pan and Zhang, Boyang},
  journal={Pattern Recognition},
  pages={110647},
  year={2024},
  publisher={Elsevier}
}
```

## Configuration Environment
```
Python3.6, Pytorch: 1.0, CUDA: 9.0+, Cudnn: follow Cuda version, GPU: one Nvidia Tesla V100
```

## Installation
First you have to make sure that you have all dependencies in place.

You can create an anaconda environment called `mgrrnet_space` using
```
conda env create -f mgrrnet_env.yaml
conda activate mgrrnet_space
```

Then, compile the extension modules.
```
python set_env_up.py build_ext --inplace
```
Then, download the BatchNet modules.

* download the [BatchNet](https://pan.baidu.com/s/1KzcgkiE-gxTy1-cw0ikaAA) via BaiDu and Extracted key is [1234]([1234]) 
* download the [BatchNet](https://drive.google.com/file/d/1fqDrqU_wMb_EbHCprZkOIWWxSiXPxaJK/view?usp=sharing) via Google

Next, put your model path in DmifNet/dmifnet/encoder/batchnet.py /def resnet18(pretrained=False, **kwargs):


## Generation
To generate meshes using a trained model, use
```
python generate.py ./config/img/mgrrnet.yaml
```

## Training
```
python train.py ./config/img/mgrrnet.yaml
```

## DataSet
You can check the baseline work Onet to download the dataset[ONet and DmifNet: DataSet](https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip). Thanks for contribution of baseline work.

## Evaluation

First, to generate meshes using a trained model, use
```
python generate.py ./config/img/mgrrnet.yaml
```

Then, for evaluation of the models, you can run it using

```
python eval_meshes.py ./config/img/mgrrnet.yaml
```
also can use quick evaluation(don't need generation).
```
python eval.py ./config/img/mgrrnet.yaml
```

## Quantitative Results
Method | Intersection over Union | Normal consistency | Chamfer distance 
:-: | :-: | :-: | :-: 
3D-R2N2 | 0.493 | 0.695 | 0.278  
Pix2Mesh | 0.480 | 0.772 | 0.216 
AtlasNet | -- | 0.811 | 0.175 
ONet | 0.571 | 0.834 | 0.215
DmifNet | 0.607 | 0.846 | 0.185
MNGDNet | 0.605 | 0.844 | 0.185
RADANet | 0.596 | 0.843 | 0.192
HFLNet | 0.613 | 0.848 | 0.182
Ours | 0.620 | 0.850 | 0.175


## Qualitative Results
<img src="display/qs.jpg" width="1000" height="500">

# Futher Information
If you have any problems with the code, please list the problems you encountered in the issue area, and I will reply you soon.
Thanks for the baseline work [Occupancy Networks - Learning 3D Reconstruction in Function Space](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks).

