# [CVPR 2026] CanonCGT: Reference-Based Color Grading via Canonical Pivot Representation.

### Jinwon Ko, Keunsoo Ko, and Chang-Su Kim.

Official code for **"CanonCGT: Reference-Based Color Grading via Canonical Pivot Representation"** in CVPR 2026. 
[[arXiv]](https://arxiv.org/pdf/2606.01638) [[paper]](https://openaccess.thecvf.com/content/CVPR2026/papers/Ko_CanonCGT_Reference-Based_Color_Grading_via_Canonical_Pivot_Representation_CVPR_2026_paper.pdf) [[video]](https://www.youtube.com/watch?v=NeXm4dkH_-k&t=5s)

<img src="https://github.com/Jinwon-Ko/CanonCGT/blob/main/assets/Overview.png" alt="overview" width="100%" height="70%" border="10"/>


## Introduction

We present **CanonCGT**, a reference-based color grading framework based on a canonical pivot representation. Our key idea is to first map the input image into a style-neutral canonical domain and then apply the reference-driven grading style from this canonical representation.

We also introduce **DP-CGT**, a dual-phase training strategy that combines supervised preset learning and self-supervised refinement for robust generalization to diverse reference images.


## Preparation

1. Installation
Create conda environment:

```bash
$ conda create -n CanonCGT python=3.9 anaconda
$ conda activate CanonCGT
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install opencv-python-headless==4.10.0.82
$ pip install pyyaml scikit-learn lpips
```

2. Pretrained models
Pretrained models will be available in:

```bash
root/CanonCGT/pretrained/
```

They can also be downloaded from [here](https://drive.google.com/file/d/1SqzCXjdJ95TAhDYY9Z4TaQPuoqlEyfkT/view?usp=sharing).


## Demo

You can run a demo with pretrained models to perform reference-based color grading on your own images.

```bash
$ cd root/CanonCGT/
$ python demo.py \
    --gpu 0 \
    --pretrained_path ./pretrained/SSL.pth \
    --inp_path SET_YOUR_INPUT_PATH \
    --ref_path SET_YOUR_REFERENCE_PATH \
    --out_path SET_YOUR_OUTPUT_PATH
```

The color-graded result will be saved to `SET_YOUR_OUTPUT_PATH`

## Dataset


## Train


## Test



## Results

Below shows our color grading results. For each pair, the left image shows the input and the right image shows the color-graded output using the inset reference image. CanonCGT produces photorealistic color grading that matches the tonal mood, lighting, and color temperature of the reference while preserving color harmony and scene structure.

<img src="https://github.com/Jinwon-Ko/CanonCGT/blob/main/assets/More_results_CanonCGT.png" alt="results" width="100%" height="70%" border="10"/>


