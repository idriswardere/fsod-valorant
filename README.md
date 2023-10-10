# Few-Shot Object Detection in Valorant using Multiplicative Layer-wise Learning Rates

**You can read the paper for this project [here](https://github.com/idriswardere/fsod-valorant/blob/master/project-report.pdf). You can also find a short presentation on the project [here](https://www.youtube.com/watch?v=UazlfWkqwDA).**

This project was heavily inspired by the work done on the ICML 2020 paper
[Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/abs/2003.06957).

![MLLR Figure](https://github.com/idriswardere/fsod-valorant/blob/master/idlspaperfig.png?raw=true)

The goal of this repository is to provide a framework for the implementation of the few-shot object detection methods in Valorant, a first-person game set in a three dimensional environment developed and published by Riot Games. We followed the few-shot object dataset detection settings in the paper.

FsDet has a two stage training scheme. In stage 1, it trained a base object detector on abundant base images and in stage 2 it add some novel pictures, fix feature extractor, retrain the final classification layer. For stage 1, we would use one of the pre-trained models published by the paper named faster rcn R 101 FPN base1. The base model we would use is built upon a ResNet-101 backbone network, which is a deep residual network architecture that has been pre-trained on the ImageNet dataset.

The custom dataset is a labeled dataset that we created for few-shot object detection in a video game. It includes 10 images of each character for the training dataset and 10 images for the validation dataset. These images were screenshots taken in the game with the character of interest in different poses and at different distances from the observer. This labeled dataset is used to finetune the pre-trained object detection model. The dataset is publicly available at the following link: https://universe.roboflow.com/fsodvalorant/fsod-valorant.


If you find this repository useful for your publications, please consider citing our code.

```angular2html
@article{Idris2023fsod-valorant,
    title={Few-Shot Object Detection in Valorant using Multiplicative Layer-wise
Learning Rates}, 
    author={Idris Wardere, Du Huang, Adrita Das, Alice Gatera},
    month = {April},
    year = {2023}
}
```

## Installation

**Requirements**

* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.4
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* CUDA 9.2, 10.0, 10.1, 10.2, 11.0
* GCC >= 4.9

**Build FsDet**
* Create a virtual environment.
```angular2html
python3 -m venv fsdet
source fsdet/bin/activate
```
You can also use `conda` to create a new environment.
```angular2html
conda create --name fsdet
conda activate fsdet
```
* Install PyTorch. You can choose the PyTorch and CUDA version according to your machine. Just make sure your PyTorch version matches the prebuilt Detectron2 version (next step). Example for PyTorch v1.6.0:
```angular2html
pip install torch==1.6.0 torchvision==0.7.0
```
Currently, the codebase is compatible with [Detectron2 v0.2.1](https://github.com/facebookresearch/detectron2/releases/tag/v0.2.1), [Detectron2 v0.3](https://github.com/facebookresearch/detectron2/releases/tag/v0.3), and [Detectron2 v0.4](https://github.com/facebookresearch/detectron2/releases/tag/v0.4). Tags correspond to the exact version of Detectron2 that is supported. To checkout the right tag (example for Detectron2 v0.3):
```bash
git checkout v0.3
```

After cloning the FSOD repository from the "Frustratingly Simple Object Detection" paper and navigating to the repo's "test" directory, you can install Detectron2 and other required packages by following these steps:
```angular2html
git clone https://github.com/idriswardere/fsod-valorant.git
```
Change your working directory to the cloned repository:
```angular2html
%cd fsod-valorant
```
* Install Detectron2 and other required packages by running the following command:.
```angular2html
!pip install git+https://github.com/facebookresearch/detectron2
!python3 -m pip install -r requirements.txt
```

## Code Structure
- **configs**: Configuration files
- **datasets**: Dataset files (see [Data Preparation](#data-preparation) for more details)
- **fsdet**
  - **checkpoint**: Checkpoint code.
  - **config**: Configuration code and default configurations.
  - **engine**: Contains training and evaluation loops and hooks.
  - **layers**: Implementations of different layers used in models.
  - **modeling**: Code for models, including backbones, proposal networks, and prediction heads.
- **tools**
  - **train_net.py**: Training script.
  - **test_net.py**: Testing script.
  - **ckpt_surgery.py**: Surgery on checkpoints.
  - **run_experiments.py**: Running experiments across many seeds.
  - **aggregate_seeds.py**: Aggregating results from many seeds.
