# EvoFill: Evolution-aware state space model for genotype imputation

![Python](https://img.shields.io/badge/python-3.10%2B-blue)

EvoFill is an advanced genotype imputation tool that leverages an evolution-aware state space model to accurately impute missing genetic variants. 



### Key Features
- **Evolution-aware modeling**: Utilizes evolutionary history to improve imputation accuracy
- **State space model**: Employs advanced statistical methods for precise genotype prediction
- **Efficient implementation**: Optimized for both speed and memory usage
- **Pre-trained weights**: Available on Hugging Face for immediate inference without training



## 1. Installation and Configuration

#### 1.1 Clone repository 

```
git clone https://github.com/MaoLab-CD/EvoFill.git
```

#### 1.2 Configuration Using Conda Virtual Environment

```bash
# Create and activate the conda environment
conda create -n evofill -f env.yml
conda activate evofill
```



## 2. Imputation 

In this tutorial, we take **human chromosome 22 (hg38)** as an example, imputation can be performed by following the steps. 

#### 2.1 **Download Pre-trained Weights**:

The following files are available and can be obtained from [Hugging Face - EvoFill](https://huggingface.co/maolab-cd/EvoFill):

- `hg38_chr22_v1.0.bin`
- `model_meta.json`
- `gt_enc_meta.json`

#### 2.2 **Prepare VCF Files for Imputation** 

Make sure your VCF files are properly formatted and ready for imputation. You will need to place the VCF files into the appropriate directory (e.g., `masked_vcf/`). 

#### 2.3 **Ensure Your File Structure is as Follows** 

Placing downloaded files in corresponding folder：

```markdown
EvoFill/
├── src/                        # EvoFill Source code
├── models/
│   ├── hg38_chr22_v1.0.bin     # Model weights
│   └── model_meta.json         # Model configuration
├── data/
│   └── masked_vcf/             # Your vcf
├── train/
│   └── gt_enc_meta.json        # Encoder metadata used during training
└── tutorial_imputation.ipynb
└── ...
```

#### 2.4 **Run Inference**:

To perform imputation,  follow the step in `tutorial_imputation.ipynb`. 



## 3. Training


1. **Partition Dataset**:
   - Split the 1000 Genomes Project dataset into training (major populations) and test (under-represented populations) sets: `notebook/1kGP_partition.ipynb`

   - Prepare 1240k-panel Sampless from multiple sources: `notebook/merge_1240k_samples.ipynb`
     - AADR (Ancient DNA Database)
     - Shimao (石峁) samples
     - Tibetan samples

2. **Model Training**: 

   - please refer to: `tutorial_training.ipynb`. 



## 4. Documentation

- **Notebooks Directory**: Contains various utility notebooks for data preparation and analysis
  - `notebook/1kGP_partition.ipynb`: Splits 1000 Genomes Project dataset into training and test sets
  - `notebook/merge_1240k_samples.ipynb`: Merges Asian and Southeastern Asian samples for augmentation
  - `notebook/Compare_summary.ipynb`: Compares imputation results
  - `notebook/Eval_summary.ipynb`: Evaluates model performance
  - `notebook/functional_region_analysis.ipynb`: Analyzes performance in functional genomic regions

- **Source Code**: Detailed implementation of the EvoFill model
  - `src/model.py`: Core model implementation
  - `src/data.py`: Data loading and preprocessing
  - `src/utils.py`: Utility functions
  - `src/eval_metrics.py`: Evaluation metrics



## 5. Performance

EvoFill has been extensively evaluated on various datasets, showing significant improvements over traditional imputation methods, particularly for under-represented populations. Detailed performance metrics are available in the project's associated publication.



## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


