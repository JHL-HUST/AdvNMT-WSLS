# Crafting Adversarial Examples for Neural Machine Translation

We provide code for the ACL2021 Paper: *Crafting Adversarial Examples for Neural Machine Translation* \
Xinze Zhang, Junzhe Zhang, Zhenhua Chen, Kun He

## Attention

The code is under maintenance to enhance its readability, and the final version will be updated in the next few weeks, as soon as possible.

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
- pkuseg, zhon, spacy, spacy([en-pipcore-web-sm](https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz#egg=en_core_web_sm==2.2.0)), spacy_cld, sacremoses, subword, pyltp (require python = 3.6),transformers (require pytorch), fairseq 
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
