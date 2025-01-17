# MNIST Digit Classification using Neural Networks

This project demonstrates the implementation of a neural network model to classify handwritten digits from the MNIST dataset. The model is built using TensorFlow and Keras and uses a fully connected feed-forward neural network (DNN) to perform the classification task.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Results](#results)


## Introduction

The MNIST dataset contains 28x28 grayscale images of handwritten digits from 0 to 9. The goal of this project is to use a neural network to classify these images with high accuracy.

We utilize TensorFlow and Keras to build the model, preprocess the data, and train the model to achieve a high classification accuracy on the test dataset. The dataset is split into training and testing sets, with the model being evaluated on the test set after training.

## Features

- **Download and preprocess the MNIST dataset**: The dataset is automatically loaded and normalized for use.
- **Build a neural network model**: A simple feed-forward neural network with two dense layers is used.
- **Train the model**: The model is trained on the MNIST training data.
- **Evaluate the model's performance**: The model is evaluated on the MNIST test set to measure accuracy and loss.
- **Visualize training progress**: Training and validation accuracy/loss are plotted to assess the model’s learning.

## Technologies Used

- **Python**: Programming language used for the project.
- **TensorFlow/Keras**: Deep learning framework to build, train, and evaluate the neural network.
- **NumPy**: For numerical computations.
- **Matplotlib**: For plotting data visualizations and model predictions.
- **MNIST Dataset**: A dataset containing 70,000 images of handwritten digits.

## Results

After training the neural network model for 100 epochs, the model achieved the following performance on the MNIST test dataset:

- **Test Accuracy**: The model achieved an accuracy of approximately **99%** on the test set, demonstrating its ability to correctly classify the handwritten digits.
- **Test Loss**: The model obtained a test loss of around **0.0700**, indicating that the predictions are close to the true labels.


