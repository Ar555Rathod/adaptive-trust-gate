# Adaptive Trust Gate

This repository contains the codebase for the **Adaptive Trust Gate** research project. The project compares traditional static hybrid recommender systems with dynamically adapting trust gates (Learned, Contextual Bandit, and GA-Evolved) to optimally balance Collaborative Filtering (CF) and Content-Based (CB) predictions based on user data sparsity.

## Repository Structure

- `src/atg/`: Core library code containing baseline experts, adaptive gates (Bandit, GA, Sequential BiLSTM), evaluation metrics, and data normalization.
- `scripts/`: Ordered execution scripts for running the full pipeline end-to-end.
- `notebooks/`: Interactive Jupyter notebooks for analysis.
- `data/`: Contains raw and processed datasets (ignored by git due to size).
- `results/`: Contains trained models, predictions, and final evaluation metrics (ignored by git due to size).

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Dataset:**
   We use the UCSD Goodreads Young Adult dataset. You will need to download the following two files and place them in `data/raw/goodreads/`:
   - `goodreads_books_young_adult.json.gz`
   - `goodreads_interactions_young_adult.json.gz`
   *(Both files are available via the official UCSD Book Graph or Kaggle).*

## Running on Google Colab (Recommended)

Due to the size of the dataset (~34 million interactions) and the computational complexity of the baseline SVD++ model and Model 7 (BiLSTM), it is highly recommended to run this pipeline on a High-RAM GPU instance in Google Colab.

1. Open a new Google Colab notebook and select a GPU runtime (T4 or L4).
2. Clone this repository directly into the Colab environment:
   ```bash
   !git clone https://github.com/Ar555Rathod/adaptive-trust-gate.git
   %cd adaptive-trust-gate
   !pip install -r requirements.txt
   ```
3. **Mount Google Drive & Link Data:**
   Since the dataset is 1.8GB and Colab deletes local files when you disconnect, you should upload the datasets to your Google Drive to avoid re-uploading them every time. 
   Upload `goodreads_books_young_adult.json.gz` and `goodreads_interactions_young_adult.json.gz` to a folder in your Drive (e.g., `MyDrive/datasets/goodreads/`).
   
   Then, in your Colab notebook, mount your Drive and copy the files into the repository:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   ```bash
   !mkdir -p data/raw/goodreads
   !cp /content/drive/MyDrive/datasets/goodreads/*.json.gz data/raw/goodreads/
   ```

4. **Run the pipeline end-to-end:**
   ```bash
   !export ATG_DATASET=goodreads && export PYTHONPATH=src && \
    python src/atg/data/normalize.py && \
    python scripts/01_build_splits.py && \
    python scripts/02_train_experts.py && \
    python scripts/03_static_hybrid.py && \
    python scripts/04_learned_gate.py && \
    python scripts/05_bandit_gate.py && \
    python scripts/06_ga_gate.py && \
    python scripts/07_sequential_gate.py && \
    python scripts/08_full_comparison.py && \
    python scripts/10_multiseed_full.py
   ```

This will automatically normalize the dataset, train all models, and spit out the final comparative metrics in `results/goodreads/metrics/full_comparison_table.csv`.
