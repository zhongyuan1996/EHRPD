# EHRPD Repository

This repository contains the implementation of **EHRPD**.

## Model Naming
- The **EHRPD model** is referred to as **MedDiffGA** in `toy.py`.
- The **PDDPM model** is defined in `unet_skip.py`.

## Paper Reference
For more details about the methodology and experimental results, please refer to the paper:

**Synthesizing Multimodal Electronic Health Records via Predictive Diffusion Models**  
[Paper Link](https://dl.acm.org/doi/abs/10.1145/3637528.3671836)

## Usage

### Creating the data

To process the data, utilize `dataPreprocess.py` on the [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/).

### Training the Model
To train the EHRPD (MedDiffGA) model, run:
```sh
python train.py
```

## Citation
If you find this repository useful for your research, please cite our paper:

@inproceedings{zhong2024synthesizing,
  title={Synthesizing multimodal electronic health records via predictive diffusion models},
  author={Zhong, Yuan and Wang, Xiaochen and Wang, Jiaqi and Zhang, Xiaokun and Wang, Yaqing and Huai, Mengdi and Xiao, Cao and Ma, Fenglong},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={4607--4618},
  year={2024}
}
