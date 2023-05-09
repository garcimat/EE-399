# EE399 HW #4

Title: Comparison of Machine Learning Algorithms: Neural Networks, SVM, Decision Tree, and LSTM on MNIST Data Set

Author: Mathew Garcia-Medina

## Abstract:

This report presents a comparison of four machine learning algorithms on the MNIST data set: feedforward neural networks, support vector machines (SVM), decision trees, and long short-term memory (LSTM). The MNIST data set is a collection of 70,000 images of handwritten digits, each of size 28x28 pixels. The goal is to correctly classify each image into one of ten classes (corresponding to the digits 0 through 9).

In this report, we first compute the first 20 principal component analysis (PCA) modes of the digit images and plot them. We then implement a feedforward neural network to classify the digits and compare its performance to the other three algorithms.

The results show that the neural network and LSTM outperformed the SVM and decision tree classifiers. The neural network achieved an accuracy of 97% on the test data set, while SVM achieved an accuracy of 97.92%, decision tree achieved an accuracy of 87.79%, and LSTM achieved an accuracy of 98.59%.

Overall, this report demonstrates the effectiveness of neural networks and LSTM on the MNIST data set and provides a comparison of different machine learning algorithms for this task.

## Sec. I. Introduction and Overview

The MNIST data set is a well-known benchmark for testing machine learning algorithms. It consists of 70,000 images of handwritten digits, each of size 28x28 pixels. The task is to classify each image into one of ten classes (corresponding to the digits 0 through 9). The MNIST data set has been widely used for testing and comparing different machine learning algorithms, and it has become a standard reference in the field.

In this report, we compare four different machine learning algorithms for classifying the MNIST data set: feedforward neural networks, support vector machines (SVM), decision trees, and long short-term memory (LSTM). First, we compute the first 20 principal component analysis (PCA) modes of the digit images and plot them. We then implement a feedforward neural network to classify the digits and compare its performance to the other three algorithms.

The report is organized as follows. In Section II, we provide a brief overview of the theoretical background of the algorithms used in this study. In Section III, we describe the algorithm implementation and development. In Section IV, we present the computational results of the study. Finally, we summarize the results and draw conclusions in Section V.

## Sec. II. Theoretical Background

In this section, we will provide a brief overview of the theoretical concepts and techniques that underlie the algorithms used in this report.

A. Linear Regression
Linear regression is a supervised learning algorithm used for predicting a continuous outcome variable based on one or more predictor variables. In the case of simple linear regression, there is only one predictor variable, while in multiple linear regression, there are two or more. The algorithm works by minimizing the sum of the squared errors between the predicted values and the actual values of the outcome variable.

B. Neural Networks
Neural networks are a class of machine learning algorithms that are inspired by the structure and function of the human brain. They are composed of interconnected nodes, or neurons, that perform computations on input data. Neural networks are used for a variety of tasks, including classification, regression, and clustering. In this report, we use feedforward neural networks for image classification.

C. Principal Component Analysis
Principal component analysis (PCA) is a technique used to reduce the dimensionality of a dataset. It works by identifying the principal components, or the directions in which the data varies the most. PCA is commonly used in image processing to reduce the dimensionality of image data while preserving as much of the variance as possible.

D. Support Vector Machines
Support vector machines (SVMs) are a class of supervised learning algorithms used for classification and regression analysis. SVMs work by finding the hyperplane that maximally separates the data into different classes. The hyperplane is chosen such that it maximizes the margin, or the distance between the hyperplane and the closest data points from each class.

E. Decision Trees
Decision trees are a type of supervised learning algorithm used for classification and regression analysis. They work by recursively partitioning the data into subsets based on the values of the predictor variables until each subset only contains data from one class. Decision trees are commonly used in data mining and machine learning due to their interpretability and ability to handle both categorical and continuous data.

## Sec. III. Algorithm Implementation and Development

In this section, we describe the algorithm implementations for the three tasks: fitting polynomials to data, PCA for MNIST image data, and classification of MNIST digits using a feed-forward neural network, LSTM, SVM, and decision tree classifiers.

Fitting Polynomials to Data
For the polynomial fitting task, we used the NumPy and Scikit-learn libraries in Python. We first loaded the data into a Pandas dataframe, and then separated it into training and test sets using the train_test_split function from Scikit-learn. We then used the training set to fit a line, parabola, and 19th degree polynomial to the data using NumPy's polyfit function. We computed the least-square error for each model over the training set and used the test set to compute the least-square error of each model. The code snippet below shows how we fit the 19th degree polynomial to the training data:

<img width="780" alt="image" src="https://user-images.githubusercontent.com/122642082/237041747-42b1a3ae-9e65-4796-b043-68118319a73d.png">

PCA for MNIST Image Data
For the PCA analysis on MNIST image data, we used the Scikit-learn library in Python. We loaded the MNIST data using Scikit-learn's fetch_openml function and then scaled the data to have zero mean and unit variance using the StandardScaler function. We then used Scikit-learn's PCA function to compute the first 20 principal components of the data. The code snippet below shows how we performed the PCA analysis:

<img width="489" alt="image" src="https://user-images.githubusercontent.com/122642082/237042218-3adcde51-ca8b-45a2-98a7-faf65ab48c04.png">

Classification of MNIST Digits
For the classification task, we used the Scikit-learn and Keras libraries in Python. We loaded the MNIST data using Scikit-learn's fetch_openml function and then split the data into training and test sets using the train_test_split function. We then used Scikit-learn's SVM and decision tree classifiers to classify the data. For the feed-forward neural network and LSTM models, we used Keras to define and train the models.

SVM Classifier
The SVM classifier was implemented using Scikit-learn's SVC class. The code snippet below shows how we trained the SVM classifier:

<img width="489" alt="image" src="https://user-images.githubusercontent.com/122642082/237043579-733adf38-d1f0-4535-9b6f-e560d6011995.png">

Finally, the accuracy of the trained SVM classifier on the test set was evaluated and printed.

For the decision tree classifier, we used Scikit-learn's DecisionTreeClassifier class. We trained the decision tree classifier on the training set and evaluated its accuracy on the test set using the following code snippet:

<img width="489" alt="image" src="https://user-images.githubusercontent.com/122642082/237043890-94cc3122-3b79-4e1d-b170-36721e6a7f02.png">

In this code, we defined a decision tree classifier with a random state of 42, trained it on the training set, and evaluated its accuracy on the test set. The accuracy of the decision tree classifier on the test set was then printed.

For the LSTM, we used Keras to build and train the model. The code snippet below shows how we defined and trained the model:

<img width="712" alt="image" src="https://user-images.githubusercontent.com/122642082/237044828-b61050fd-f56f-40e3-a3b4-c1c25e4bc3b8.png">

For the feed-forward neural network, we used PyTorch to build and train the model. The code snippet below shows how we defined and trained the model:

<img width="597" alt="image" src="https://user-images.githubusercontent.com/122642082/237045096-855c224a-4105-4518-af4b-02ccb68ec3f8.png">

## Sec. IV. Computational Results

In this section, we present the computational results obtained from implementing the feedforward neural network, LSTM, SVM, and decision tree classifiers on the MNIST dataset.

Below are plots of the first 20 PCA components from the MNIST dataset:

### Figure 1. Plots of First 20 PCA Components from MNIST Dataset
<img width="627" alt="image" src="https://user-images.githubusercontent.com/122642082/237047014-b81304cd-d8d0-47f3-a2ce-2208b6aecf42.png">

For the feedforward neural network, we achieved an accuracy of 97% on the test set. The code snippet above shows how we trained and tested the neural network and the code snippet below shows the results:

<img width="589" alt="image" src="https://user-images.githubusercontent.com/122642082/237045444-ee74ad3b-a285-4bfb-ac2d-34633cb9bf1d.png">

For the LSTM classifier, we achieved an accuracy of 98.59% on the test set. The code snippet below shows the results:

<img width="1130" alt="image" src="https://user-images.githubusercontent.com/122642082/237046080-6dfe6947-11cc-4473-9220-6dfd8549d3f4.png">

For the SVM classifier, we achieved an accuracy of 97.92% on the test set. The code snippet below shows how we trained and tested the SVM classifier:

<img width="271" alt="image" src="https://user-images.githubusercontent.com/122642082/237046245-4e0de17f-b120-4cfc-9939-d64db6a1893c.png">

For the decision tree classifier, we achieved an accuracy of 87.79% on the test set. The code snippet below shows the results:

<img width="271" alt="image" src="https://user-images.githubusercontent.com/122642082/237046331-ad007dda-6ac7-4bb8-b474-3c12cab0aa32.png">

In summary, we achieved an accuracy of 97% with the feed-forward neural network, 97.92% with the SVM classifier, 87.79% with the decision tree classifier, and 98.59% with the LSTM classifier on the MNIST dataset.

## Sec. V. Summary and Conclusions

In this project, we implemented and compared several machine learning algorithms for classification of the MNIST handwritten digits dataset.

In the first part of the project, we applied principal component analysis (PCA) to reduce the dimensionality of the dataset and visualize the first 20 PCA modes of the digit images. We then trained a feed-forward neural network on the MNIST data set and achieved an accuracy of 97% on the test set.

In the second part, we compared the performance of the neural network against three other classifiers: SVM, decision tree, and LSTM. The SVM classifier achieved an accuracy of 97.92%, the decision tree achieved an accuracy of 87.79%, and the LSTM achieved an accuracy of 98.59%. These results show that the LSTM classifier performed the best on this dataset, followed closely by the neural network and SVM classifiers.

Overall, this project demonstrates the effectiveness of machine learning algorithms for classification tasks, particularly for handwritten digit recognition. Future work could involve exploring more advanced neural network architectures or using other dimensionality reduction techniques in combination with machine learning algorithms.
