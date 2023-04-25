# EE399 HW #3
Author: Mathew Garcia-Medina

## Abstract:

In this project, we analyze the MNIST dataset, a large collection of handwritten digits, to explore the effectiveness of machine learning algorithms in classifying this data. We begin by performing exploratory data analysis and dimensionality reduction using PCA to gain insights into the dataset's characteristics. We then implement linear discriminant analysis (LDA), support vector machines (SVM), and decision tree classifiers to classify two sets of digits: those that are easiest to separate and those that are hardest to separate. We compare the accuracy and training time of each algorithm to determine which is the most effective in classifying the data.

Our results show that all three classifiers perform well in classifying the digits, with SVM being the most accurate overall. However, the LDA and decision tree classifiers were also effective in separating the digits, with high accuracy rates and relatively short training times. These findings highlight the importance of selecting the appropriate algorithm for the task at hand and demonstrate the potential of machine learning in analyzing and classifying large datasets such as MNIST.

## Sec. I. Introduction and Overview

The MNIST dataset is a widely used benchmark in machine learning, consisting of 60,000 training images and 10,000 test images of handwritten digits. The goal of this project is to analyze the MNIST dataset and implement several classification algorithms to determine which is most effective in classifying the data.

In this report, we explore the effectiveness of linear discriminant analysis (LDA), support vector machines (SVM), and decision tree classifiers in classifying two sets of digits from the MNIST dataset: those that are easiest to separate (digits 2 and 9) and those that are hardest to separate (digits 4 and 9). We compare the accuracy and training time of each algorithm to determine which is the most effective in classifying the data.

In Section II, we provide an overview of the theoretical background and concepts relevant to our analysis, including dimensionality reduction, LDA, SVM, and decision tree classifiers. In Section III, we describe the implementation and development of our algorithms and discuss the hyperparameters used for each algorithm. In Section IV, we present the computational results, including the accuracy rates and training times for each algorithm. Finally, in Section V, we summarize our findings and discuss the implications of our results.

Overall, this project aims to provide a comprehensive analysis of the MNIST dataset and demonstrate the potential of machine learning algorithms in analyzing and classifying large datasets.

## Sec. II. Theoretical Background

The MNIST dataset consists of 28x28 grayscale images of handwritten digits, with each image represented as a vector of 784 dimensions. Due to the high dimensionality of the dataset, it is often necessary to perform dimensionality reduction to reduce computational complexity and improve accuracy.

Linear discriminant analysis (LDA) is a technique used for dimensionality reduction and classification. LDA finds a linear combination of features that maximizes the separation between classes while minimizing the variance within each class. This can be used to project high-dimensional data onto a lower-dimensional subspace while preserving the discriminatory information between classes.

Support vector machines (SVM) are a popular supervised learning method used for classification and regression analysis. SVM finds the optimal hyperplane that separates the data into different classes. SVM can be used for both linearly separable and non-linearly separable data by mapping the data into a higher-dimensional space using kernel functions.

Decision tree classifiers are a type of supervised learning method used for classification and regression analysis. Decision trees are constructed by recursively splitting the data into smaller subsets based on the values of the features. The goal is to create a tree that minimizes the impurity of the subsets at each node, resulting in a hierarchy of if-then rules that can be used for classification.

In this project, we will use LDA, SVM, and decision tree classifiers to classify two sets of digits from the MNIST dataset: digits 2 and 9, which are easiest to separate, and digits 4 and 9, which are hardest to separate. By comparing the performance of each algorithm in terms of accuracy and training time, we aim to determine which algorithm is most effective in classifying the data.

## Sec. III. Algorithm Implementation and Development

### Data Preprocessing

The MNIST dataset consists of 28 x 28 grayscale images of handwritten digits. Before applying any machine learning algorithms, the images need to be preprocessed to extract useful features and convert them into a format that can be fed into the algorithms. In this project, we will be using two digits, 1 and 8, to test the performance of LDA, SVM, and decision tree classifiers.

We begin by loading the necessary libraries and the MNIST dataset.

<img width="532" alt="image" src="https://user-images.githubusercontent.com/122642082/234194473-9b613c09-076d-4d94-b035-bf2e8a33f975.png">

Next, we extract the images and labels of the two digits we are interested in and split the dataset into training and testing sets. i.e., 1 and 4.

<img width="691" alt="image" src="https://user-images.githubusercontent.com/122642082/234194758-acc34f38-6564-4a6e-92da-ee4fd0dc8df3.png">

### Linear Discriminant Analysis (LDA)

LDA is a linear classification algorithm that tries to find the linear combination of features that best separates the classes. It does this by maximizing the ratio of between-class variance to within-class variance.

We begin by importing the necessary libraries and creating an instance of the LDA model. Next, we fit the model to the training data and transform the data into the LDA space. We can now train a classifier on the transformed data and evaluate its performance on the test set.

<img width="335" alt="image" src="https://user-images.githubusercontent.com/122642082/234195299-7596547b-6d19-43ec-a42e-5da3d0b52b47.png">

### Support Vector Machine (SVM)

SVM is a powerful classification algorithm that works by finding the hyperplane that best separates the classes. It can handle high-dimensional data and is effective even when the number of features is greater than the number of samples.

We begin by importing the necessary libraries and creating an instance of the SVM model. Next, we fit the model to the training data and evaluate its performance on the test set.

<img width="684" alt="image" src="https://user-images.githubusercontent.com/122642082/234195756-54ebb748-5cc1-47fe-ad6c-786bf95645af.png">

### Decision Tree Classifier

Decision trees are a popular classification algorithm that work by recursively splitting the data based on the features that best separate the classes. They are easy to interpret and can handle both categorical and numerical data.

We begin by importing the necessary libraries and creating an instance of the decision tree classifier.

<img width="690" alt="image" src="https://user-images.githubusercontent.com/122642082/234196008-807ea13c-223c-49ec-a922-6644f33e4568.png">

## Sec. IV. Computational Results

### Singular Value Spectrum Plot:

#### Figure 1. Plot of SVP

<img width="535" alt="image" src="https://user-images.githubusercontent.com/122642082/234198209-b4f49464-22d4-4859-b027-6d46fbc45edf.png">

### Interpretations of U, S, and V Matrices:

#### Figure 2. Interpretations of Matrix Parameters

<img width="612" alt="image" src="https://user-images.githubusercontent.com/122642082/234198325-4e514fdd-c21a-431d-9405-bcb40324672d.png">

### Plots of First 4 Principal Components:

#### Figure 3. Visualization of First Principal Components

<img width="691" alt="image" src="https://user-images.githubusercontent.com/122642082/234198491-1bbe7f1b-94ea-4dfd-8be5-b74b467f8866.png">

### 3D Plot of Projected Data (on 2, 3, 5):

#### Figure 4. Plot of Projected Data

<img width="377" alt="image" src="https://user-images.githubusercontent.com/122642082/234198660-15ffd257-93f3-4840-9523-1a340d931938.png">

### Accuracy Values for Digits:

First, for the LDA model, we achieved high accuracy for most digit pairs. For digits 1 and 4, the model achieved an accuracy of 0.99. For digits 1, 3, and 8, the accuracy was perfect at 1.0. For digits 4 and 9, the model achieved an accuracy of 0.96, while for digits 2 and 9, the accuracy was 0.99. However, for all 10 digits, the LDA model achieved an accuracy of 0.868, indicating that it struggled with classifying some of the more challenging digit pairs.

Next, we evaluated the performance of the SVM model on the same digit pairs. For digits 4 and 9, the SVM model achieved an accuracy of 0.984, while for digits 2 and 9, the accuracy was 0.983. When classifying all 10 digits, the SVM model achieved an overall accuracy of 0.976, which is higher than the accuracy achieved by the LDA model.

Finally, we evaluated the decision tree classifier's performance on the same digit pairs. For digits 4 and 9, the model achieved an accuracy of 0.934, while for digits 2 and 9, the accuracy was 0.977. For all 10 digits, the decision tree classifier achieved an accuracy of 0.870, which is lower than the accuracy achieved by both the LDA and SVM models. A table with all the data can be seen below.

#### Figure 5. Table with Accuracy Data with Different Digits

<img width="641" alt="image" src="https://user-images.githubusercontent.com/122642082/234201175-4f3ed56f-4a11-43c6-8f79-a3375023cbbb.png">


## Sec. V. Summary and Conclusions

In this project, we analyzed the MNIST dataset containing images of handwritten digits and tested three different classifiers - LDA, SVM, and Decision Tree Classifier - to see which algorithm can classify the data most accurately.

We started by performing SVD analysis of the digit images to understand the rank of the digit space, and then built linear classifiers for two digits that were easiest and hardest to separate. We then built a Support Vector Machine and a Decision Tree Classifier to classify all ten digits.

We found that SVM performed the best in classifying all ten digits, with an overall accuracy of 0.976. LDA also performed well, achieving an accuracy of 1 when classifying digits 1, 3, and 8. However, the DTC performed poorly in classifying all ten digits, with an overall accuracy of 0.870.

Based on our results, we can conclude that SVM is the best classifier for the MNIST dataset, achieving the highest overall accuracy. However, LDA can also be a good choice for certain digit classification tasks. The poor performance of DTC in classifying all ten digits suggests that it may not be the best algorithm to use for this type of image classification problem.

In conclusion, our findings suggest that SVM is the most suitable algorithm for classifying handwritten digits in the MNIST dataset, and it can be used for various image recognition applications.
