# [CVPR 2026] CanonCGT: Reference-Based Color Grading via Canonical Pivot Representation.

### Jinwon Ko, Keunsoo Ko, and Chang-Su Kim.

Official code for **"CanonCGT: Reference-Based Color Grading via Canonical Pivot Representation"** in CVPR 2026. 
[[arXiv]](https://arxiv.org/pdf/2606.01638) [[paper]](https://openaccess.thecvf.com/content/CVPR2026/papers/Ko_CanonCGT_Reference-Based_Color_Grading_via_Canonical_Pivot_Representation_CVPR_2026_paper.pdf) [[video]](https://www.youtube.com/watch?v=NeXm4dkH_-k&t=5s)

<img src="https://github.com/Jinwon-Ko/CanonCGT/blob/main/assets/Overview.png" alt="overview" width="100%" height="70%" border="10"/>


## 📝 Introduction

We present **CanonCGT**, a reference-based color grading framework based on a canonical pivot representation. Our key idea is to first map the input image into a style-neutral canonical domain and then apply the reference-driven grading style from this canonical representation.

We also introduce **DP-CGT**, a dual-phase training strategy that combines supervised preset learning and self-supervised refinement for robust generalization to diverse reference images.


## ⚙️ Preparation

### 1. Installation

Create conda environment:

```bash
$ conda create -n CanonCGT python=3.9 anaconda
$ conda activate CanonCGT
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install opencv-python-headless==4.10.0.82
$ pip install pyyaml scikit-learn lpips
```

### 2. Pretrained models

Pretrained models will be available in:

```bash
root/CanonCGT/pretrained/
```

They can also be downloaded from [here](https://drive.google.com/file/d/1SqzCXjdJ95TAhDYY9Z4TaQPuoqlEyfkT/view?usp=sharing).


## 🚀 Demo

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


## 📂 Dataset

We use both supervised paired data and unsupervised image collections for training CanonCGT.

### 1. Supervised Paired Dataset

For supervised preset learning, we construct a paired dataset from the **MIT-Adobe FiveK** dataset.  
Specifically, we use the Expert C version as the canonical target and generate preset-transformed images using Lightroom presets.

Due to license restrictions, we do **not** redistribute the Lightroom presets or the generated paired dataset.  
Instead, we provide the source links to the presets used in our experiments, so that users can download them directly from the original providers and follow their respective license terms.

- [MIT-Adobe FiveK Dataset](https://data.csail.mit.edu/graphics/fivek/)
- [Lightroom preset source links](./assets/lightroom_preset_links.md)

### 2. Unsupervised Dataset

For self-supervised refinement, we use diverse unpaired image datasets from multiple domains.

Please download each dataset from its official source:

- [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- [LSDIR](https://github.com/ofsoundof/LSDIR)
- [PPR10K](https://github.com/csjliang/PPR10K)
- [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)


## 🏋️ Train
CanonCGT is trained using **DP-CGT**, a dual-phase training strategy consisting of supervised preset learning and self-supervised refinement.

### Phase 1: Supervised Preset Learning

In Phase 1, CanonCGT is trained using the supervised paired dataset constructed from MIT-Adobe FiveK and Lightroom presets.  
This phase learns the canonical pivot representation and the reference-based grading process from preset-based paired data.

#### Phase 1-A: Grade Extractor Training

First, train the grade extractor to encode tonal characteristics from preset-transformed images.

```bash
$ cd root/CanonCGT/
$ python main.py \
    --gpu 0 \
    --yaml Stage1_style_encoder \
    --host server \
    --run_mode train
```

After training the grade extractor, save the style centroids:

```bash
$ python main.py \
    --gpu 0 \
    --yaml Stage1_style_encoder \
    --host server \
    --run_mode test \
    --load
```

#### Phase 1-B: Canonicalizer and Grader Training

Next, train the canonicalizer and the grader using the supervised paired dataset.

```bash
$ python main.py \
    --gpu 0 \
    --yaml Stage1_canonicalizer \
    --host server \
    --run_mode train
```

```bash
$ python main.py \
    --gpu 0 \
    --yaml Stage1_styler \
    --host server \
    --run_mode train
```

#### Phase 1-C: End-to-End Fine-tuning

Then, fine-tune the full CanonCGT framework in an end-to-end manner.

```bash
$ python main.py \
    --gpu 0 \
    --yaml Stage2_end_to_end_finetuning \
    --host server \
    --run_mode train
```

### Phase 2: Self-Supervised Refinement

In Phase 2, CanonCGT is further refined using unpaired image datasets.  
This phase improves the generalization ability of CanonCGT to diverse real-world reference images beyond the preset-based supervised training data.

```bash
$ python main.py \
    --gpu 0 \
    --yaml Stage3_SSL_training_Flickr2K_PPR10K_LSDIR \
    --host server \
    --run_mode train
```

Alternatively, you can run the full training pipeline with:

```bash
$ bash run.sh
```


## 🧪 Evaluation

You can evaluate CanonCGT using a pretrained or trained model with:

```bash
$ python main.py \
    --gpu 0 \
    --yaml Stage3_SSL_training_Flickr2K_PPR10K_LSDIR \
    --host server \
    --run_mode eval \
    --load \
    --viz
```

The evaluation results and visualized outputs will be saved to the output directory specified in the corresponding configuration file.


## 🎨 Results

The figure below shows our color grading results. For each pair, the left image shows the input and the right image shows the color-graded output using the inset reference image. CanonCGT produces photorealistic color grading that matches the tonal mood, lighting, and color temperature of the reference while preserving color harmony and scene structure.

<img src="https://github.com/Jinwon-Ko/CanonCGT/blob/main/assets/More_results_CanonCGT.png" alt="results" width="100%" height="70%" border="10"/>


