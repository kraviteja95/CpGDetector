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
8. [Few Points to Ponder](#points-to-ponder)

## Introduction

The code consists of several components:

- **Dataset Preparation:** Random DNA sequences are generated, and the number of CpG sites in each sequence is counted to create the dataset.
- **Model Architecture:** The CpG predictor model is implemented using an LSTM neural network followed by a fully connected layer.
- **Training:** The model is trained using the prepared dataset.
- **Evaluation:** The trained model is evaluated on a separate test dataset.
- **Variable-Length Sequences:** Support for variable-length DNA sequences is added, where sequences can have lengths within a certain range.
- **Saving the Model:** The trained model is saved to disk for future use.

## Dataset Preparation

- The dataset consists of randomly generated DNA sequences, where each sequence is associated with the count of CpG sites it contains.
- The `rand_sequence` function generates random DNA sequences, and the `count_cpgs` function counts the number of CpG sites in each sequence.
- The training and test datasets are prepared using these functions. Note that `prepare_data` is the function used for dataset preparation.
- When calling the prepare_data function, you pass `2048` as the argument for training data, indicating that you want to prepare a dataset with `2048` samples for training the model.
- Similarly, `512` is passed as the argument for testing data, indicating that you want to prepare a dataset with `512` samples for testing the model's performance.

## Model Architecture

The model architecture is defined by the `CpGPredictor` class, which is a subclass of `torch.nn.Module`. Let's look at the overview of the model architecture:

- **LSTM Layer**: The model begins with an LSTM (Long Short-Term Memory) layer, which is responsible for processing input sequences and capturing temporal dependencies. The input to the LSTM layer is a sequence of DNA bases, where each base is represented by a single value. The parameters of the LSTM layer are:
  - `input_size=1`: Each element in the input sequence is a single value (representing a DNA base).
  - `hidden_size=LSTM_HIDDEN`: The number of features in the hidden state of the LSTM.
  - `num_layers=LSTM_LAYER`: The number of recurrent layers.
  - `batch_first=True`: The input and output tensors are provided as (batch_size, sequence_length, input_size) by default.

- **Fully Connected Layer**: Following the LSTM layer, there is a fully connected (linear) layer. This layer takes the output of the LSTM layer and maps it to a single output, which represents the predicted number of CpG sites in the input sequences.

The `forward` method of the `CpGPredictor` class defines the forward pass of the model. It specifies how input data flows through the layers defined in the `__init__` method. The input tensor is reshaped to match the input requirements of the LSTM layer, passed through the LSTM layer, and then through the fully connected layer to obtain the final prediction.

This architecture creates a simple model that takes DNA sequences as input, processes them using an LSTM layer to capture sequential information, and predicts the number of CpG sites in the input sequences.

## Training

The model is trained using the prepared dataset. The training loop iterates over batches of sequences and their corresponding labels. The sequences are padded to ensure uniform length within each batch. The `Mean Squared Error` loss function is used, and the Adam optimizer is employed for optimization.

## Evaluation

After training, the model is evaluated on a separate test dataset. The evaluation loop follows a similar structure to the training loop, where batches of sequences are processed, and predictions are made. The Mean Absolute Error metric is calculated to assess the model's performance.

## Variable-Length Sequences

Support for variable-length sequences is added by modifying the dataset preparation process. The `rand_sequence_var_len` function generates sequences with lengths within a specified range. No padding is required for variable-length sequences during training and evaluation.

## Saving the Model

The trained model is saved to disk in two formats: as a state dictionary (`trained_model.pth`) and as the entire model (`cpg_detector_model.pth`). These files can be downloaded and used for inference on new data.

## Few Points to Ponder

## Why Did We Use LSTM with PyTorch?

We used LSTM (Long Short-Term Memory) with PyTorch because it's well-suited for sequential data like DNA sequences. LSTMs have the ability to capture long-range dependencies in sequences, making them effective for tasks like sequence prediction.

### Comparison with Other Options

- **TensorFlow**: TensorFlow also offers LSTM implementation, but historically, PyTorch has been favored for its simplicity and flexibility in research and development settings.
- **Transformers**: Transformers, popularized by models like BERT, are another option for sequence modeling. They can capture long-range dependencies efficiently using attention mechanisms. However, LSTMs are still widely used for tasks where sequential information is crucial.

## Why Did We Use Flask?

We used Flask for serving the trained model as an API because it's lightweight, simple, and well-suited for building RESTful APIs. Flask allows us to define routes and handle HTTP requests easily.

### Comparison with Other Options

- **Streamlit**: Streamlit is more focused on creating interactive data applications and dashboards with minimal coding. While it's great for data visualization and exploration, it may not be as suitable for building APIs for serving machine learning models.
- **Django**: Django is a full-fledged web framework with built-in features for authentication, ORM, and admin interface. It's more heavyweight compared to Flask and may be overkill for simple API deployments.

## Detailed Explanations and Comparisons

- **Dataset Preparation**: We generated synthetic DNA sequences and labeled them based on the number of CpG sites. This approach allows us to train a model on simulated data, which can generalize well to real-world scenarios.
- **Training and Evaluation**: We used PyTorch's training framework to train the model on the prepared dataset. The model was optimized using the Adam optimizer and evaluated using metrics like Mean Absolute Error. This approach ensures that our model learns to make accurate predictions.
- **API Deployment**: We deployed the trained model as an API using Flask, which allows external applications to interact with the model for making predictions. Flask's lightweight nature and simplicity make it suitable for deploying small-scale machine learning models.
