# Crafting Adversarial Examples for Neural Machine Translation

We provide code for the ACL2021 Paper: *Crafting Adversarial Examples for Neural Machine Translation* \
Xinze Zhang, Junzhe Zhang, Zhenhua Chen, Kun He

## Environment
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

### Pre-installation
Due to the space limitation of the github, some auxiliary files, especially the pre-trained model files and the processed word vector files, are too big to be uploaded in the current repository.

And the pre-trained BERT models, i.e., the uncased whole word masked models, used in this work, had the effective web-link when this work was just published. But these links are not available directly for now (2022 Apr.) becasue of some unknown reasons. To tackle this issue, you need search the model name in https://huggingface.co/models .

Hence, for the replication, we provide these necessary files in the Google Drive. You need download these files as follows and copy them to the corresponding path (cop_path) to finish the pre-installation procedure.

|File_name| Cop_path             | Link|    Remark            |
|-----------------|--------------------|:--------:|--------|
| en_core_web_sm-2.2.0.tar.gz | ./aux_files/en_core_web_sm-2.2.0.tar.gz | [click here](https://drive.google.com/file/d/1i29J4gPAInJwNA1cVdhQWZKdkra41ts9/view?usp=sharing) | Provided by Spacy.          |
| googleNewsWV.bin.vectors.npy.tar.gz         | ./aux_files/googleNewsWV.bin.vectors.npy            | [click here](https://drive.google.com/file/d/1C7edw_4pNZNgrCchGhS_wnjFGB5KUrXD/view?usp=sharing) | Pre-trained by Google. It's a file, unzip first.     |
| en-de-en.tar.gz             | ./models/Transformer/en-de-en           | [click here](https://drive.google.com/file/d/1ugFZBRjRqbn7wZpiORG2dmZe_ANLyX5O/view?usp=sharing) | Pre-trained by FAIR. It's a folder, unzip first. |
| jobs.tar.gz                 | ./corpus/wmt19/jobs                     | [click here](https://drive.google.com/file/d/1UPokk3xDv2vbaz21nceczehBdNw4x1wD/view?usp=sharing) | It's a folder, unzip first. |
| test.en                     | ./corpus/wmt19/test.en                  | [click here](https://drive.google.com/file/d/1wuVr7bNocGASkoyx77nteE3wkaERJ4_J/view?usp=sharing) | Provided by WMT19.         |
| test.de                     | ./corpus/wmt19/test.de                    | [click here](https://drive.google.com/file/d/1e5xkUxkcfM8Ci0vrzp0A2oeIHpfZ9FgY/view?usp=sharing) | Provided by WMT19.         |


Due to the space limitation of Google Drive and the huge size of the NMT models as well as word-vector models, we only upload the model files related to en-de experiment. For the model files about other experiments, please contact the code [contributor](https://github.com/XinzeZhang) of this repository.

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

## Usage

- Conduct the pre-installation and installation precedures to download the necessary files as well as configure the dependencies.
- Use `corpus/wmt19/split.py` to split the test set into many small job sets to accelerate the experiments with executing main file on different small jobs in parallel (you can also download the job sets directly as introduced in pre-installation).
- Run `main_rorr.py` or use command like `python main_rorr.py -job 0`, where `-job 0` can be replaced to any job id x, such as `-job x`, and `main_rorr.py` can be replaced to `main_rogr.py` and `main_gogr.py` to conduct the specific attack method.
- Before runing `main_wsls.py`, `main_gogr.py` need be executed first to initialize the adversarial example, then executing the shell script `bash dumped/data_move.sh` to copy the GOGR results to the right place for runing the rest of the proposed WSLS method. After that, run `main_wsls.py` or use command like `python main_wsls.py -job 0`, where `-job 0` can be replaced to any job id x, such as `-job x`.
- The pre-trained BERT model used in this work would be auto-downloaded to the cache folder when the code is executed at first time.


## Contact
For replication, if you have any questions of the code details, please contact xinze (xinze@hust.edu.cn), the first author of this paper.

--2022-04-10--

The code is under maintenance to enhance its readability.