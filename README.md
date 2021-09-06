# Edge and Identity Preserving Network for Face Super-Resolution (EIPNet)     - Neurocomputing Journal 2021 

## Abstract

<details>
  <summary> CLICK ME </summary>
Face super-resolution (SR) has become an indispensable function in security solutions such as video surveillance and identification system, but the distortion in facial components is a great challenge in it. Most state-of-the-art methods have utilized facial priors with deep neural networks. These methods require extra labels, longer training time, and larger computation memory. In this paper, we propose a novel Edge and Identity Preserving Network for Face SR Network, named as EIPNet, to minimize the distortion by utilizing a lightweight edge block and identity information. We present an edge block to extract perceptual edge information, and concatenate it to the original feature maps in multiple scales. This structure progressively provides edge information in reconstruction to aggregate local and global structural information. Moreover, we define an identity loss function to preserve identification of SR images. The identity loss function compares feature distributions between SR images and their ground truth to recover identities in SR images. In addition, we provide a luminance-chrominance error (LCE) to separately infer brightness and color information in SR images. The LCE method not only reduces the dependency of color information by dividing brightness and color components but also enables our network to reflect differences between SR images and their ground truth in two color spaces of RGB and YUV. The proposed method facilitates the proposed SR network to elaborately restore facial components and generate high quality 8x scaled SR images with a lightweight network structure. Furthermore, our network is able to reconstruct an 128x128 SR image with 215 fps on a GTX 1080Ti GPU. Extensive experiments demonstrate that our network qualitatively and quantitatively outperforms state-of-the-art methods on two challenging datasets: CelebA and VGGFace2.
</details>

> Edge and Identity Preserving Network for Face Super-Resolution    
> Jonghuyn Kim, Gen Li, Inyong Yun, Cheolkon Jung, Joongkyu Kim    
> **Neurocomputing Journal 2021**

[[Paper](https://www.sciencedirect.com/science/article/pii/S0925231221004227)]

## Installation

Clone this repo.

Install requirements:
```
  tensorflow==1.14.0
  numpy
  matplotlib
  opencv-python
```

## Dataset

This network is pretrained on [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. After downloading this dataset, unzip and save test images in a `./datasets` folder. 

## Generating HR images using a pretrained model

After preparing test images, the reconstructed images can be obtained using the pretrained model.

1. Creat a `checkpoint/CelebA` folder. Download pretrained weight from [Google Drive](https://drive.google.com/file/d/1393OZ8ZIShFQi3IA18meqokFan0zRjm4/view?usp=sharing) and upzip this `checkpoint.zip` in the `./checkpoint/CelebA` folder.
2. Run `test.py` to generate HR images, which will be saved in `./checkpoint/CelebA/result`. Save path and details can be edited in `./options/base_options.py` and `./options/test_options.py`.

## Training a new model on personal dataset
We update `train.py` to train EIPNet on personal dataset.

1. Save train and test images in `./datasets/train` and `./datasets/test` folders, respectively.
2. Check your personal setting (i.e., implementation details, save path, and so on) in `./options/base_options.py` and `./options/train_options.py`.
3. Run `train.py` or type 'python train.py' in your terminal.

## License
All right reserved. Licensed under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode) (Attribution-NonCommercial-NoDerivatives 4.0 International). The code is released for academic research use only.

## Citation
If you use this code for your research, please cite our papers.


