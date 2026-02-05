# EvoFill: Evolution-aware state space model for genotype imputation

![GitHub](https://img.shields.io/github/license/yourusername/EvoFill)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

## 📋 Project Overview

EvoFill is an advanced genotype imputation tool that leverages an evolution-aware state space model to accurately impute missing genetic variants. By incorporating evolutionary information into the imputation process, EvoFill achieves superior performance, especially for under-represented populations.

### Key Features
- **Evolution-aware modeling**: Utilizes evolutionary history to improve imputation accuracy
- **State space model**: Employs advanced statistical methods for precise genotype prediction
- **Population-specific optimization**: Performs particularly well for under-represented populations
- **Efficient implementation**: Optimized for both speed and memory usage
- **Pre-trained weights**: Available on Hugging Face for immediate inference without training
- **Easy to use**: Simple API for both inference and training

## 🚀 Installation and Configuration

### Using Conda Virtual Environment
```bash
# Create and activate the conda environment
conda create -n evofill -f env.yml
conda activate evofill
```

### Using Docker
```bash
# Build the Docker image
docker build -t evofill .

# Run the container
docker run -it --rm evofill
```

## 📁 Dataset Preparation

### Required Only for Training from Scratch
Dataset preparation is only necessary if you plan to retrain the model. When using pre-trained weights from Hugging Face, **no training datasets are required**.

If you need to retrain the model, follow these steps:

1. **Partition 1kGP Dataset**:
   - Split the 1000 Genomes Project dataset into training (major populations) and test (under-represented populations) sets
   - Use the provided notebook: `notebook/1kGP_partition.ipynb`

2. **Prepare 1240k-panel Samples**:
   - Merge Asian and Southeastern Asian samples from multiple sources:
     - AADR (Ancient DNA Database)
     - Shimao (石峁) samples
     - Tibetan samples
   - Use the provided notebook: `notebook/merge_1240k_samples.ipynb`

## 📖 Tutorial

### Using Pre-trained Weights (Recommended)
EvoFill can directly use pre-trained weights from Hugging Face for inference, eliminating the need for training from scratch:

1. **Download Pre-trained Weights**:
   - Access the model weights on Hugging Face: [EvoFill Model](https://huggingface.co/yourusername/evofill)
   - Follow the instructions to download and load the weights

2. **Run Inference**:
   - Use the pre-trained model directly for genotype imputation
   - No training dataset required

### Training from Scratch
If you need to retrain the model with custom data, follow the comprehensive tutorial notebook:

- **Main Tutorial**: `tuorial.ipynb`
  - Step-by-step guide for model training
  - Imputation workflow
  - Result evaluation and interpretation

## 📚 Documentation

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

## 🔬 Usage Example

### Using Pre-trained Weights (Inference Only)
```python
# Example code snippet for using EvoFill with pre-trained weights
from evofill import EvoFillModel

# Initialize the model with pre-trained weights
model = EvoFillModel.from_pretrained("yourusername/evofill")

# Perform imputation directly (no training required)
imputed_genotypes = model.impute(test_data_path)

# Save results
model.save_results(imputed_genotypes, output_path)
```

### Training from Scratch
```python
# Example code snippet for training EvoFill
from evofill import EvoFillModel

# Initialize the model
model = EvoFillModel()

# Train the model
model.train(training_data_path)

# Save the trained model
model.save("path/to/save/model")

# Perform imputation
imputed_genotypes = model.impute(test_data_path)

# Save results
model.save_results(imputed_genotypes, output_path)
```

## 📈 Performance

EvoFill has been extensively evaluated on various datasets, showing significant improvements over traditional imputation methods, particularly for under-represented populations. Detailed performance metrics are available in the project's associated publication.

## 🤝 Contributing

We welcome contributions to EvoFill! If you're interested in contributing, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License

This project is licensed under the MIT License.


