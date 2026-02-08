# EvoFill: Evolution-aware state space model for genotype imputation

![Python](https://img.shields.io/badge/python-3.10%2B-blue)

EvoFill is an advanced genotype imputation tool that leverages an evolution-aware state space model to accurately impute missing genetic variants. 



### Key Features
- **State space model**: Built on Mamba-2 architecture with linear complexity
- **Multi-panel compatibility**: Supports various sequencing panels (WGS, 1240k, etc.)
- **Evolution-aware loss**: Incorporates evolutionary information for better imputation
- **Public weights**: Pre-trained weights available without requiring original genetic data


## 1. Installation and Configuration

#### 1.1 Clone repository 

```
git clone https://github.com/MaoLab-CD/EvoFill.git
```

#### 1.2 Configuration Using Conda Virtual Environment

EvoFill is developed under the following environment:
```
Python ver:     3.10.19
Pytorch ver:    2.8.0
CUDA ver:       12.9
Mamba ver:      2.2.5
Deepspeed ver:  0.17.5
```

For complete environment configuration:  
```bash
# Create and activate the conda environment
conda create -n evofill -f env.yml
conda activate evofill
```


## 2. Imputation 

In this tutorial, we take **human chromosome 22 (hg38)** as an example, imputation can be performed by following the steps. 

#### 2.1 **Download Pre-trained Weights**:

The following files are available and can be obtained from [Hugging Face - EvoFill](https://huggingface.co/maolab-cd/EvoFill/tree/main/hg38_chr22):

- `hg38_chr22_v1.0.bin`: Model weights
- `model_meta.json`: Model configuration
- `gt_enc_meta.json`: Encoder metadata 


#### 2.2 **Prepare VCF Files for Imputation** 

Make sure your VCF files are properly formatted and ready for imputation. You will need to place the VCF files into the appropriate directory (e.g., `input_vcf/`). 

#### 2.3 **Ensure File Structure** 

Placing downloaded files in corresponding folder：

```markdown
EvoFill/
├── src/                        # Source code in GitHub
├── models/
│   ├── hg38_chr22_v1.0.bin
│   ├── model_meta.json
│   └── gt_enc_meta.json
├── data/
│   └── input_vcf/              # Your vcf
└── tutorial_imputation.ipynb
└── ...
```

#### 2.4 **Run Inference**:

To perform imputation,  follow the step in **[tutorial_imputation.ipynb](tutorial_imputation.ipynb)**. 



## 3. Training

To reproduce the 2-stage training protocol in our paper, please follow the steps:

1. **Dataset Partition**:
   - Split the 1000 Genomes Project (1kGP) dataset into training (dominate populations) and test (underrepresented population, CDX in our case) sets: **[notebook/1kGP_partition.ipynb](notebook/1kGP_partition.ipynb)**

   - Prepare 1240k-panel Sampless from multiple sources: **[notebook/merge_1240k_samples.ipynb](notebook/merge_1240k_samples.ipynb)**
     - AADR (Ancient DNA Database)
     - Shimao Archaeological Site (石峁遗址) samples
     - Modern Tibetan samples

2. **2-stage training**: 

   - please refer to: **[tutorial_training.ipynb](tutorial_training.ipynb)**. 


## 4. Documentation

- **Source Code**: Implementation of the EvoFill backbone
  - `src/model.py`: EvoFill backbone implementation
  - `src/data.py`: Data loading and preprocessing
  - `src/utils.py`: Utility functions
  - `src/eval_metrics.py`: Evaluation metrics

- **Notebooks**: notebooks for analysis in our paper
  - `notebook/1kGP_partition.ipynb`: Splits 1000 Genomes Project dataset into training and test sets
  - `notebook/merge_1240k_samples.ipynb`: Merges 1240k-panel samples for augmentation
  - `notebook/Eval_summary.ipynb`: Evaluates model performance
  - `notebook/functional_region_analysis.ipynb`: Analyzes performance in functional genomic regions


## 5. Cite

*under review*


## 6. License

This project is licensed under the MIT License 

