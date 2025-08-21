# EvoFill: Evolutionary Trajectory-Informed Genotype Imputation

**内部开发中**

Pingcode：https://pchz20250707025859383.pingcode.com/ship/products/JYX

GitHub 仓库不包含`./data`，`./ckpt` 等数据文件夹

完整本地项目请见：`192.168.10.5:/mnt/qmtang/EvoFill/`

## Installation and configuration

```bash
conda create -f environment.yml
conda activate evofill
```

## Dataset preparation

```bash
python src/vcf2tensor.py
```

## Training

```bash
deepspeed train.py --deepspeed config/ds_config.json
```

## Imputation

```bash
python imputation.py
```

## Results visualization 

```bash
jupyter notebook ...
```