# CpG Predictor using PyTorch

This repository contains code for a simple CpG (Cytosine-phosphate-Guanine) predictor implemented using PyTorch. The model uses a Long Short-Term Memory (LSTM) neural network to count the number of CpG sites in a DNA sequence.

## Contents

1. [Introduction](#introduction)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Variable-Length Sequences](#variable-length-sequences)
7. [Saving the Model](#saving-the-model)

## Introduction

The code consists of several components:

- **Dataset Preparation:** Random DNA sequences are generated, and the number of CpG sites in each sequence is counted to create the dataset.
- **Model Architecture:** The CpG predictor model is implemented using an LSTM neural network followed by a fully connected layer.
- **Training:** The model is trained using the prepared dataset.
- **Evaluation:** The trained model is evaluated on a separate test dataset.
- **Variable-Length Sequences:** Support for variable-length DNA sequences is added, where sequences can have lengths within a certain range.
- **Saving the Model:** The trained model is saved to disk for future use.

## Dataset Preparation

The dataset consists of randomly generated DNA sequences, where each sequence is associated with the count of CpG sites it contains. The `rand_sequence` function generates random DNA sequences, and the `count_cpgs` function counts the number of CpG sites in each sequence. The training and test datasets are prepared using these functions.

## Model Architecture

The CpG predictor model is implemented as a PyTorch module named `CpGPredictor`. It consists of an LSTM layer followed by a fully connected layer. The LSTM layer processes the input sequences, and the fully connected layer produces the final prediction.

## Training

The model is trained using the prepared dataset. The training loop iterates over batches of sequences and their corresponding labels. The sequences are padded to ensure uniform length within each batch. The Mean Squared Error loss function is used, and the Adam optimizer is employed for optimization.

## Evaluation

After training, the model is evaluated on a separate test dataset. The evaluation loop follows a similar structure to the training loop, where batches of sequences are processed, and predictions are made. The Mean Absolute Error metric is calculated to assess the model's performance.

## Variable-Length Sequences

Support for variable-length sequences is added by modifying the dataset preparation process. The `rand_sequence_var_len` function generates sequences with lengths within a specified range. No padding is required for variable-length sequences during training and evaluation.

## Saving the Model

The trained model is saved to disk in two formats: as a state dictionary (`trained_model.pth`) and as the entire model (`cpg_detector_model.pth`). These files can be downloaded and used for inference on new data.
