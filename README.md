# Music Genre Classification

This project implements three different approaches to music genre classification using machine learning and deep learning techniques.

[Genre Classification Notebook](./genre_classification-SiddharthSaxena_AndreasPapaeracleous.ipynb)
or
<a target="_blank" href="https://colab.research.google.com/github/SidSaxena01/Genre-Classification/blob/main/genre_classification-SiddharthSaxena_AndreasPapaeracleous.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[Mood Classification Notebook](./mood_classification-SiddharthSaxena.ipynb)
or
<a target="_blank" href="https://colab.research.google.com/github/SidSaxena01/Genre-Classification/blob/main/mood_classification-SiddharthSaxena.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Project Overview

Music genre classification is a fundamental task in Music Information Retrieval (MIR) that automatically categorizes music tracks based on their audio characteristics. We explore three different methodologies for classifying the GTZAN dataset into 10 genres (Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, and Rock).

## Dataset

The data is provided in three formats:
1. **Essentia descriptors** (ARFF files): Audio features extracted using the Essentia library
2. **Global and segment features** (CSV files): Features extracted at different time scales (30-second and 3-second)
3. **Spectrograms** (PNG images): Visual representations of audio frequency content over time

## Task 1: Essentia Descriptors Classification

We used audio descriptors provided in ARFF format to train and evaluate various machine learning models.

### Implementation Details:
- Processed ARFF files, handling 3 categorical features and 237 numerical features
- Applied feature selection using ANOVA F-test to identify the most informative features
- Compared multiple algorithms (Logistic Regression, SVM, KNN, Decision Trees, Random Forest, Neural Networks)
- Performed hyperparameter tuning for top-performing models
- Implemented ensemble learning using a voting classifier

### Results:
- **Neural Network**: 100% test accuracy (best model)
- **SVM**: 94.8% test accuracy
- **Logistic Regression**: 90.7% test accuracy
- **Ensemble**: 99.5% test accuracy

### Key Observations:
- Perfect test accuracy suggests either exceptional feature separation or potential overfitting
- Notable gap between cross-validation (83.1%) and test performance (100%)
- Essentia descriptors provide highly discriminative features for genre classification

## Task 2: CSV Features Classification

We compared global features (30-sec) and segment-level features (3-sec) provided in CSV format to determine which time scale is more effective.

### Implementation Details:
- Applied similar methodology to Task 1 (feature selection, model comparison, hyperparameter tuning)
- Created separate pipelines for 30-second and 3-second features
- Conducted comparative analysis between the two feature sets

### Results:
- 3-second features generally achieved higher accuracy
- Similar model ranking pattern to Task 1
- Feature importance analysis revealed key audio characteristics for genre discrimination

## Task 3: CNN with Spectrograms

We implemented deep learning approaches to classify music genres directly from spectrogram images.

### Implementation Details:
- Loaded and preprocessed spectrogram images (224×224 RGB)
- Applied data augmentation (rotation, flips, color adjustments)
- Implemented two models:
  - **Custom CNN**: 4 convolutional blocks with increasing filter complexity
  - **Audio Spectrogram Transformer (AST)**: Transformer-based architecture for spectrograms
- Used PyTorch
- Implemented early stopping and learning rate scheduling

### Results:
- **Custom CNN**: 76.0% test accuracy
- **AST**: 60.5% test accuracy
- Most confusion between country/rock (25%) and reggae/hiphop (20%)

### Key Observations:
- CNN architecture excelled at capturing local spectrogram patterns
- Simpler CNN outperformed the more complex AST model
- Genre confusion patterns aligned with musicological similarities
- CNN's local pattern recognition proved more effective than the transformer's global attention

## Comparative Analysis

| Approach | Best Model | Test Accuracy |
|----------|------------|----------|
| Essentia Descriptors | Neural Network | 100.0% |
| CSV Features | Varied by feature set | 80-90% |
| Spectrograms | Custom CNN | 76.0% |

## Installation Requirements

> Using uv (faster, modern Python package installer)

`uv sync` or
`uv pip sync requirements.txt`

> Using traditional pip

`pip install -r requirements.txt`

Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
torch
torchvision
opencv-python
tqdm
```

## Running the Code

1. Ensure all data files are in the correct directories (data/):
   - `GenreTrain.arff` and `GenreTest.arff` for Task 1
   - `features_30_sec.csv` and `features_3_sec.csv` for Task 2
   - Spectrogram images in `images_original/` directory for Task 3

2. Run each task in sequence in the notebook:

3. Review the results and visualizations within the notebook

## Conclusion

This project demonstrates that while hand-crafted audio features (Essentia descriptors) currently provide the best classification performance, deep learning approaches with spectrograms show promise, especially with domain-specific architectures. The custom CNN's superior performance over the AST model highlights that sometimes simpler architectures better capture the relevant patterns for music genre classification.