# GRPE: Relative Positional Encoding for Graph Transformer

Official implementation of [GRPE](https://arxiv.org/abs/2201.12787). 
We achieve the second best model on the [PCQM4Mv2 dataset of the OGB-LSC Leaderboard](https://ogb.stanford.edu/docs/lsc/leaderboards/).


## Quick Start

### Prepair environment
```bash
conda env create --file environment.yaml
conda activate chemprop
```

### Prepare pretrained weight
Download pretrained weights from ```https://drive.google.com/drive/folders/1Oc3Ox0HAoJ5Hrihfp5-jFvStPIfFQAf9?usp=sharing```
and create folder ```pretrained_weight```.


## Reproduce results

Please check ```{dataset-name}.sh``` for detailed commands to reproduce the results.


## Hardware requirements

* 4 gpus (A100 with 80GiB) are required to run experiments for PCQM4M, PCQM4Mv2, PCBA and HIV.
* 1 gpu is required to run experiments for MNIST and CIFAR10.


## Molecule Finger-Print

### Installation
```bash
pip install git+https://github.com/lenscloth/GraphSelfAttention

# Install pytorch & pytorch geometric version according to your environment
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cpu.html
```

### Example
```python
from grpe.pretrained import load_pretrained_fingerprint

fingerprint_model = load_pretrained_fingerprint(cuda=True)
finger = fingerprint_model.generate_fingerprint(
    [
        "CC(=O)NCCC1=CNc2c1cc(OC)cc2",
    ],
    fingerprint_stack=5,
) # 1x3840 Pytorch Tensor

```

