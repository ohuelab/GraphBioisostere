# GraphBioisostere

A Graph Neural Network-based model for predicting bioisosteric replacement in drug discovery. This repository contains the implementation of a deep learning approach that predicts whether molecular transformations will maintain or improve biological activity across different protein targets.

## Overview

Bioisosteric replacement is a crucial strategy in medicinal chemistry for optimizing drug candidates. This project implements a graph neural network model that learns to predict the viability of matched molecular pair (MMP) transformations using molecular graph representations.

### Key Features

- **Graph-based molecular representation**: Uses PyTorch Geometric for efficient graph neural network operations
- **Target-aware learning**: Handles multiple protein targets with shared molecular representations
- **Multiple prediction modes**: Supports whole-molecule and fragment-based predictions
- **Transfer learning**: Fine-tuning capabilities for target-specific predictions
- **Baseline comparisons**: Includes GBDT (Gradient Boosting Decision Trees) baseline using ECFP4 fingerprints

## Repository Structure

```
GraphBioisostere/
├── pro_GNN/              # Main GNN implementation
│   ├── config.py         # Model configuration
│   ├── training_cls_ddp_3.py  # Main training script (DDP support)
│   ├── finetune_reg.py   # Transfer learning script
│   ├── prep_cls_all.py   # Data preparation for whole molecules
│   ├── prep_cls_frag_all.py  # Data preparation for fragments
│   ├── encoder/          # GNN encoder implementations
│   ├── model/            # Model architectures
│   ├── utils/            # Utility functions
│   └── local/            # Local execution scripts
├── gbdt/                 # GBDT baseline implementation
│   ├── prep_ecfp4.py     # ECFP4 fingerprint generation
│   ├── prep_ecfp4_frag.py  # Fragment ECFP4 generation
│   └── run_gbdt.py       # GBDT training and evaluation
├── splitting/            # Data splitting utilities
│   └── data_splitting.py # Target-based 5-fold CV splitting
├── notebooks/            # Jupyter notebooks for analysis
│   ├── data.ipynb        # Data exploration and selection
│   └── data_process.ipynb  # Data preprocessing
├── test/                 # Testing and evaluation scripts
│   └── generate_final_figures.py  # Figure generation for results
└── README.md            # This file
```

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.12+
- PyTorch Geometric
- RDKit
- LightGBM (for baseline)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GraphBioisostere.git
cd GraphBioisostere
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install rdkit pandas numpy scikit-learn lightgbm matplotlib seaborn tqdm
```

## Data Preparation

### Download Dataset

The dataset should be downloaded separately and placed in a directory (e.g., `MMP_dataset/`). The dataset should contain:

- `dataset_consistentsmiles.csv`: Main CSV file with MMP pairs and labels
- The CSV should include columns: `smiles1`, `smiles2`, `label`, `tid` (target ID), etc.

### Data Structure

The expected CSV format:
```
index,smiles1,smiles2,frag1,frag2,tid,label,delta_value
0,CC(C)Cc1ccc(cc1)C(C)C(O)=O,CC(C)Cc1ccc(cc1)C(C)C(=O)NO,C(C)C(O)=O,C(C)C(=O)NO,1,1,2.5
```

### Generate PyTorch Dataset

Convert CSV to PyTorch geometric data format:

```bash
cd pro_GNN
python prep_cls_all.py /path/to/MMP_dataset/
```

This generates `dataset_consistentsmiles.pt` containing graph representations of all molecule pairs.

### Data Splitting

Create 5-fold cross-validation splits based on target IDs:

```bash
cd splitting
python data_splitting.py \
    --csv_path /path/to/MMP_dataset/dataset_consistentsmiles.csv \
    --output_dir /path/to/pro_GNN/dataset/dataset_cv \
    --data_path /path/to/MMP_dataset/dataset_consistentsmiles.pt \
    --pkl_output tid_5cv.pkl \
    --seed 41
```

This creates:
- `dataset_cv1.pt`, `dataset_cv2.pt`, ..., `dataset_cv5.pt`: Train/val/test splits for each fold
- `tid_5cv.pkl`: Metadata for the splits

## Training

### GNN Model Training

Train the graph neural network model with distributed data parallel (DDP):

```bash
cd pro_GNN/local
bash run_ddp.sh 'dataset/dataset_cv1.pt' 'results/cv1/pair_diff' 'pair' 'diff'
```

Parameters:
- `dataset_cv1.pt`: Input dataset file
- `results/cv1/pair_diff`: Output directory
- `pair`: Prediction mode ('pair', 'frag', etc.)
- `diff`: Loss type ('diff', 'cat', 'product')

For all 5 folds:
```bash
for i in {1..5}; do
    bash run_ddp.sh "dataset/dataset_cv${i}.pt" "results/cv${i}/pair_diff" 'pair' 'diff'
done
```

### Fragment-based Training

For fragment-only predictions:

```bash
python prep_cls_frag_all.py /path/to/MMP_dataset/
# Then split and train similarly
```

### GBDT Baseline

Train LightGBM baseline with ECFP4 fingerprints:

```bash
cd gbdt
# Generate fingerprints
python prep_ecfp4.py /path/to/MMP_dataset/ --radius 2 --n_bits 2048

# Train GBDT
python run_gbdt.py
```

## Evaluation

Test predictions are automatically saved during training:
- `results/cv*/*/test_predictions.npz`: Contains predictions, true labels, and indices

To generate evaluation metrics and figures:

```bash
cd test
python generate_final_figures.py
```

## Transfer Learning

Fine-tune the pre-trained model on a specific target:

```bash
cd pro_GNN
python finetune_reg.py \
    --pretrained_model results/cv1/pair_diff/best_model.pth \
    --target_data notebooks/target/target_data.pt \
    --output_dir results/target/finetune
```

## Model Configuration

Key hyperparameters in `pro_GNN/config.py`:

```python
class Args:
    batch_size = 8192       # Batch size
    hidden_dim = 64         # Hidden dimension
    embedding_dim = 64      # Embedding dimension
    num_layers = 2          # Number of GNN layers
    dropout = 0.2           # Dropout rate
    lr = 1e-3              # Learning rate
    epochs = 500           # Maximum epochs
    patience = 30          # Early stopping patience
```

## Results

The trained models produce:
- **Classification metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, MCC
- **Prediction files**: NPZ files with predictions for analysis
- **Training logs**: Loss curves and validation metrics

Results are organized by:
- Cross-validation fold (cv1-cv5)
- Prediction mode (pair, frag, etc.)
- Loss type (diff, cat, product)

## Citation
Masunaga S, Furui K, Kengkanna A, Ohue M. **GraphBioisostere: general bioisostere prediction model with deep graph neural network**. _J Supercomput_ 82, 132 (2026). https://doi.org/10.1007/s11227-026-08232-y

## License
Apache-2.0 license

## Contact
For questions or issues, please:
- Open an issue on GitHub
- Contact: 



