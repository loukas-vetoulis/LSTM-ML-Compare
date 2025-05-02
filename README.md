# Sentiment Analysis Model Comparison

This project implements and compares multiple machine learning approaches for sentiment analysis on text reviews:
1. Custom and scikit-learn implementations of Naïve Bayes and Logistic Regression algorithms
2. Deep learning approach using a Bidirectional LSTM model with PyTorch

The implementation focuses on hyperparameter optimization and performance evaluation using learning curves and precision-recall metrics.

## Project Structure

- **main.py**: Executes the hyperparameter optimization process for both Naïve Bayes and Logistic Regression algorithms.
- **main_firstPart.py**: Runs custom implementations of the classification models with optimal hyperparameters and generates precision tables.
- **main_secondPart.py**: Applies the optimal hyperparameters to scikit-learn's implementations and outputs precision tables for comparison.
- **Utils.py**: Contains utility functions for text vectorization and feature extraction.
- **Kampiles.py**: Implements learning curve computation functionality to assess model performance across varying training set sizes.
- **Kampiles_run.py**: Executes learning curve calculations for both custom and scikit-learn implementations.
- **bilstm_model.py**: Implements a deep learning approach using a stacked Bidirectional LSTM neural network with PyTorch.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/sentiment-analysis-comparison.git
cd sentiment-analysis-comparison
pip install -r requirements.txt
```

## Requirements

- Python 3.6+
- tqdm
- torch
- tensorflow
- scikit-learn
- matplotlib
- numpy
- keras

## Usage

### Hyperparameter Optimization

To run the hyperparameter optimization process:

```bash
python main.py
```

This will perform grid searches for both algorithms:
- Logistic Regression: optimizes learning rate (0.001-0.1) and regularization parameter λ (0.001-0.1)
- Naïve Bayes: optimizes the Laplace smoothing parameter (0.1-5.0)

### Custom Model Evaluation

To evaluate the custom implementations with optimal hyperparameters:

```bash
python main_firstPart.py
```

### Scikit-learn Model Evaluation

To evaluate the scikit-learn implementations with the same optimal hyperparameters:

```bash
python main_secondPart.py
```

### Learning Curves

To generate learning curves comparing both implementations:

```bash
python Kampiles_run.py
```

## Optimal Hyperparameters

Based on extensive grid search experiments, the following optimal hyperparameters were identified:

### Logistic Regression
- Training sample size: 6000
- Learning rate: 0.011
- Regularization parameter (λ): 0.001

### Naïve Bayes
- Training sample size: 6000
- Laplace smoothing parameter: 0.9

## Feature Selection

Both models utilize information gain for feature selection, identifying the most informative words from the dataset to improve model efficiency.

## Results

The project compares the performance of:
1. Custom implementations against scikit-learn's implementations of traditional machine learning algorithms
2. Traditional machine learning approaches versus deep learning approaches

Results are presented through precision tables and learning curves for comprehensive analysis.

## BiLSTM Deep Learning Model

The project includes an implementation of a Stacked Bidirectional LSTM model using PyTorch for sentiment analysis.

### Model Architecture
- Embedding layer with dimension 100
- 2-layer Bidirectional LSTM with hidden dimension 128
- Global max pooling layer
- Fully connected layer with sigmoid activation

### Training Setup
- Dataset: IMDB movie reviews (loaded via Keras)
- Vocabulary size: 10,000 most frequent words
- Sequence length: 500 tokens (padded)
- Optimizer: Adam with learning rate 0.001
- Loss function: Binary Cross Entropy
- Batch size: 32
- Training strategy: Early stopping with patience of 3 epochs

### Usage

To train and evaluate the BiLSTM model:

```bash
python bilstm_model.py
```

The script will:
1. Load and preprocess the IMDB dataset
2. Split data into train/dev/test sets
3. Train the BiLSTM model with early stopping
4. Evaluate performance on the test set
5. Generate a loss curve plot and print precision/recall metrics

## Contact

Loukas Vetoulis - [GitHub Profile](https://github.com/loukas-vetoulis)

Project Link: [https://github.com/loukas-vetoulis/C-SGG-Asteroids-Arcade](https://github.com/loukas-vetoulis/C-SGG-Asteroids-Arcade)

