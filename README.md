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

Clone the github repo and set up conda environment 

```
# Clone the GitHub Repository
$ git clone <this-repo-url>

# Create a new environment
$ conda create --name agile python=3.9 -y
$ conda activate agile

# Install PyTorch and torchvision with CUDA support. Make sure the versions are compatible with your CUDA version.
$ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113  --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install torch-geometric==2.2.0 torch-sparse==0.6.16 torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
$ pip install -r requirements.txt
```

### Dataset

We have provided the fine-tuning library and candidate library used in the paper in `data.zip`, extract the zip file under `./data` folder. 

### Pre-training

The pre-trained AGILE model on the 60k virtual lipid library can be found in `ckpt/pretrained_agile_60k`. If your data significantly differs from the 60k virtual lipid library, the pre-trained AGILE model might not perform optimally. In such cases, you can pre-train the model with your own dataset to potentially achieve better results.

Steps to Pre-train with Your Data:
1. **Obtain Pre-trained Base Models**: Download the pre-trained [MolCLR](https://www.nature.com/articles/s42256-022-00447-x) models, which serve as a starting point for further training. These models are available at [here](https://github.com/yuyangw/MolCLR).

1. **Set Up the Model Directory**:
Place the downloaded MolCLR model files in the `./ckpt` directory within your project folder. This ensures they are properly accessed by the training script.

1. **Configure Training Settings**:
Open the `config_pretrain.yaml` file and make the following adjustments:
- `load_model`: Change this to the model name of your downloaded MolCLR model.
- `data_path`: Specify the path to your dataset where the training data is stored.

```
$ python pretrain.py config_pretrain.yaml
```


### Fine-tuning



To fine-tune the AGILE pre-trained model for ionizable lipid prediction on the specific cell lines, you can modify the configurations in `config_finetune.yaml`. We have provided the pre-trained AGILE model on the 60k virtual lipid library, which can be found in `ckpt/pretrained_agile_60k`.

```
$ python finetune.py config_finetune.yaml
```

### Inference and visualization

The current 'infer_vis.py' will perform model inference with the AGILE fine-tuned model on the fine-tuning dataset. To perform inference on new data, you can modify the config file to specify the new data path.

```
$ python infer_vis.py <folder name of the fine-tuned model>
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
## Acknowledgement

- MolCLR: [https://github.com/yuyangw/MolCLR](https://github.com/yuyangw/MolCLR)
- Mordred: [https://github.com/mordred-descriptor/mordred](https://github.com/mordred-descriptor/mordred)
