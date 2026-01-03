# EvoFill: Evolutionary Trajectory-Informed Genotype Imputation

**内部开发中**

Pingcode：https://pchz20250707025859383.pingcode.com/ship/products/JYX

完整本地项目请见：`192.168.10.5:/mnt/qmtang/EvoFill/`

## Installation and configuration

```bash
conda create -n evofill -f env.yml
conda activate evofill
```

## Dataset preparation

1. split training and validation set
```bash
bash scripts/split_vcf.sh
```

2. calculate the p-distance among samples:
```bash
Vcf2Dis -InPut training_vcf.gz -OutPut training.p_dis.mat
Vcf2Dis -InPut val_vcf.gz -OutPut val.p_dis.mat
```

3. Convert vcf to tensor
```bash
python src/vcf2tensor.py
```

## Training

1. config the model and training process

2a. strat training in local
```bash
deepspeed train.py --deepspeed config/ds_config.json
```

2b. strat training in cluster
```bash
deepspeed train.py --deepspeed config/ds_config.json
```

3. monitor the training
```bash
[todo]
```

## Imputation

```bash
python imputation.py
```

## Results visualization 

```bash
jupyter notebook ...
```