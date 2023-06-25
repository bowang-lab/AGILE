# AGILE

This is the official codebase for **AGILE Platform: A Deep Learning-Powered Approach to Accelerate LNP Development for mRNA Delivery**.[[biorXiv]](https://www.biorxiv.org/content/10.1101/2023.06.01.543345v1.abstract)

## Introduction

AGILE (**A**I-**G**uided **I**onizable **L**ipid **E**ngineering) platform streamlines the iterative development of ionizable lipids, crucial components for LNP-mediated mRNA delivery. This platform brings forth three significant features: 

:test_tube: Efficient design and synthesis of combinatorial lipid libraries\
:brain: Comprehensive in silico lipid screening employing deep neural networks\
:dna: Adaptability to diverse cell lines

It also significantly truncates the timeline for new ionizable lipid development, reducing it from potential months or even years to weeks :stopwatch:！

An overview of AGILE can be seen below:

<p align="center">
  <img src="https://github.com/bowang-lab/AGILE/blob/590b980e55a4e43dff5f1bc8c86d2d02791be05e/figures/AGILE_overview.png" alt="AGILE architecture diagram" border="0">
</p>


## Getting Started

### Installation

Set up conda environment and clone the github repo

```
# create a new environment
$ conda create --name agile python=3.9 -y
$ conda activate agile

# install requirements
$ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113  --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install torch-geometric==2.2.0 torch-sparse==0.6.16 torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
$ pip install -r requirements.txt
```
### Pre-training

To pre-train AGILE with your own data, you can modify the configurations in `config_pretrain.yaml`. The pretraining is inherited from 
```
$ python pretrain.py config_pretrain.yaml
```

### Fine-tuning

To fine-tune the AGILE pre-trained model for ionizable lipid prediction on the specific cell lines, you can modify the configurations in `config_finetune.yaml`. We have provided the pre-trained AGILE model on the 60k virtual lipid library, which can be found in `ckpt/pretrained_agile_60k`.

```
$ python finetune.py config_finetune.yaml
```

## Citing AGILE

```bibtex
@article{xu2023agile,
  title={AGILE Platform: A Deep Learning-Powered Approach to Accelerate LNP Development for mRNA Delivery},
  author={Xu, Yue and Ma, Shihao and Cui, Haotian and Chen, Jingan and Xu, Shufen and Wang, Kevin and Varley, Andrew and Lu, Rick Xing Ze and Bo, Wang and Li, Bowen},
  journal={bioRxiv},
  pages={2023--06},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
