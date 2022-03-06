# Crafting Adversarial Examples for Neural Machine Translation

We provide code for the Paper: *Crafting Adversarial Examples for Neural Machine Translation* \
Xinze Zhang, Junzhe Zhang, Zhenhua Chen, Kun He

## Introduction

Effective adversary generation for neural machine translation (NMT) is a crucial prerequisite for building robust machine translation systems. In this work, we investigate veritable evaluations of NMT adversarial attacks, and propose a novel method to craft NMT adversarial examples. We propose to leverage the round-trip translation technique to build valid metrics for evaluating NMT adversarial attacks. Our intuition is that an effective NMT adversarial example, which imposes minor shifting on the source and degrades the translation dramatically, would naturally lead to a semantic-destroyed round-trip translation result. We then propose a promising black-box attack method called Word Saliency speedup Local Search (WSLS) that could effectively attack the mainstream NMT architectures.

Our main contributions are as follows:

1) We introduce an appropriate definition of NMT adversary and the deriving evaluation metrics, which are capable of estimating the adversaries only using source information, and tackle well the challenge of missing ground-truth reference after the perturbation.

2) We propose a novel black-box word level NMT attack method that could effectively attack the mainstream NMT models, and exhibit high transferability when attacking popular online translators.


## Server Environment
- Pos_OS 20.04.LTS
- Python 3.6
- CUDA >= 10.2
  - use `cat /usr/lib/cuda/version.txt` to check the cuda version
- NCCL for fairseq
  - The official and tested builds of NCCL can be downloaded from: https://developer.nvidia.com/nccl
- apex for faster training (require RTX-based GPUs)
  ```
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
  ```

## Requirements
```
pickle
sklearn
torch
nltk
pkuseg
transformers
pyltp
zhon
spacy
spacy(en-pipcore-web-sm)
spacy_cld
sacremoses
subword-nmt
fairseq
fvcore
```
### Installation
- pickle
  - default package in the protogenetic conda python env
- sklearn, nltk, torch
  ```
  conda install scikit-learn
  conda install nltk
  conda install pytorch cudatoolkit=10.2 -c pytorch
  ```
- pkuseg, zhon, spacy, spacy([en-pipcore-web-sm](https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz#egg=en_core_web_sm==2.2.0)), spacy_cld, sacremoses, subword, pyltp (require python = 3.6),transformers (require pytorch), fairseq (require Pytorch >= 1.4.0, Python >= 3.5, Nvidia GPU , and NCCL)
  ```
  pip install pkuseg
  pip install zhon
  pip install sacremoses
  pip install subword-nmt
  pip install pyltp
  pip install transformers
  pip install fvcore
  pip install spacy-cld
  pip install -U spacy
  pip install ./aux_files/en_core_web_sm-2.2.0.tar.gz

  cd ..
  git clone https://github.com/pytorch/fairseq
  cd fairseq
  pip install --editable ./
  ```
  where  `en_core_web_sm-2.2.0.tar.gz` can be downloaded in the [link](https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz#egg=en_core_web_sm==2.2.0)
  