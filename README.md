# Image-Classification-With-and-Without-NN(CIFAR-10)

In this project, I tried different methods to classify images from the CIFAR-10 dataset. The dataset has 60,000 color images in 10 different classes like airplanes, cars, birds, cats, and more. Each image is small (32x32 pixels), and there are 6,000 images for each class.Both classic machine learning methods and deep learning are used to compare results.

### Models and Accuracy:

* HOG + PCA + SGD → **47%**
* HOG + Color Histogram + PCA + XGBoost → **60%**
* HOG + PCA + SVC → **62%**
* Data Augmentation + CNN → **81%**

---

## Overview 

### Overview of HOG and PCA

**HOG (Histogram of Oriented Gradients):** HOG is a technique used to describe how an image looks. It focuses on the edges and shapes in small parts of the image. It checks how the brightness changes and saves the direction of those changes (gradients). This helps to find patterns like outlines of animals or machines.

**PCA (Principal Component Analysis):** PCA is a method to reduce the number of features. When HOG gives many numbers to describe an image, PCA picks only the most important ones. This keeps the main information and removes noise or repeated data. It also helps the model to learn faster and overfit less.

### Combining HOG and PCA:

* **Feature Extraction:** First, HOG is applied to each image to get a feature vector (a list of numbers showing gradients).
* **Dimensionality Reduction:** Then PCA is used to reduce the size of these vectors.
* **Classification:** The smaller set of features is used by a classifier like SVC, SGD, or XGBoost to predict the image class.

### Benefits of Combining HOG and PCA:

* **Better Speed:** PCA reduces how much data we use, so training is faster.
* **Less Noise:** PCA helps remove unimportant or repeated features.
* **Better Generalization:** The model may perform better on new data because it learns only the most useful patterns.

### What is Color Histogram?

Color histogram is a way to describe the color information in an image. It counts how many pixels are red, green, or blue at different levels. This gives a general idea of what colors are in the image and how often they appear. When combined with HOG, it helps the model understand both the shape and color of objects.

---

## CNN with Data Augmentation

CNN (Convolutional Neural Network) is a deep learning model that automatically learns features from images. It uses special filters to detect edges, textures, and shapes.

Data augmentation is used to make the model better. It creates new images by slightly changing the original ones (rotating, flipping, zooming, etc.). This helps the CNN learn better and not just memorize the training images. This method gave the best accuracy: **81%**.

---

## Why was SGD not accurate?

SGD (Stochastic Gradient Descent) is a  linear model, so it could not separate the complex patterns in CIFAR-10 images. This lack of nonlinearity made it less accurate compared to methods like SVC or XGBoost. That’s why its accuracy was low: **47%**.

---

## Model Comparison:

* Classical models with HOG, PCA, and color histograms gave moderate results (around 60% accuracy).
* CNN with data augmentation gave the best result (81% accuracy).

In general, the models could tell apart animals and machines well. But similar animals or similar vehicles were sometimes mixed up.
The most common confusion was between:
* **Cat and Dog**
* **Horse and Deer**
* **Truck and Car**

---
## Summary

The results show that traditional machine learning models do not perform as well as neural networks, however with techniques like HOG, PCA, and color histograms, they can still achieve decent accuracy and be useful for specific datasets.
