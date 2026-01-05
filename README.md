# EvoFill: Evolutionary Trajectory-Informed Genotype Imputation

## Installation and configuration

with conda virtual environment:
``` bash
conda create -n evofill -f env.yml
conda activate evofill
```

with docker images:
``` bash
docker build -t evofill .
```

## Dataset preparation

1. partition the 1kGP dataset into major population (training) and under-represented population (test) set: `notebook/1kGP_partition.ipynb`

2. merge Asian and Eastsourthen-Asian samples in AADR, Shimao(石峁) and Tibitan sample together as `1240k-panel` sample for augmentation: `notebook/merge_1240k_samples.ipynb`

## Tutorial for model training and imputation

please following: `tuorial.ipynb`

## Imputation

``` bash
python imputation.py
```

## Cite

