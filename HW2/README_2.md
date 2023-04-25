# EE399 HW #1
Author: Mathew Garcia-Medina

## Abstract:

This Python implementation explores polynomial regression as a method for predicting data points. Three polynomial models with increasing degrees (linear, quadratic, and 19th degree) are fit to a sample dataset, and the training and testing errors for each model are calculated. Additionally, the models are fit to different subsets of the data to test their predictive power. The implementation uses the numpy and matplotlib libraries for data manipulation and visualization.

## Sec. I. Introduction and Overview

Polynomial regression is a method of curve fitting used to model the relationship between a dependent variable and one or more independent variables. The goal is to find the polynomial function that best fits the data points, so that the function can be used to predict future observations. In this implementation, we explore polynomial regression using a sample dataset and compare the performance of different polynomial models.

## Sec. II. Theoretical Background

Polynomial regression involves fitting a polynomial equation of degree n to a set of data points. The general form of a polynomial of degree n is:

y = a_0 + a_1x + a_2x^2 + ... + a_n*x^n

where y is the dependent variable, x is the independent variable, and a_0, a_1, ..., a_n are the coefficients of the polynomial. The coefficients are estimated using a method called least squares, which minimizes the sum of the squared differences between the predicted and actual values of the dependent variable.

## Sec. III. Algorithm Implementation and Development

The implementation consists of the following steps:

Loading the sample dataset

Fitting linear, quadratic, and 19th degree polynomials to the data

Calculating the training and testing errors for each model

Fitting the models to different subsets of the data to test their predictive power

Visualizing the results using matplotlib


Three different algorithms for fitting polynomials to data points:

1. A linear regression algorithm was used to fit a line to the first 20 data points.

2. A quadratic regression algorithm was used to fit a parabola to the first 20 data points.

3. A high-degree regression algorithm was used to fit a 19th degree polynomial to the first 20 data points.

We also implemented an additional set of algorithms for fitting polynomials to a subset of the data points, which consisted of the first 10 and last 10 points:

1. A linear regression algorithm was used to fit a line to the first and last 10 data points.

2. A quadratic regression algorithm was used to fit a parabola to the first and last 10 data points.

3. A high-degree regression algorithm was used to fit a 19th degree polynomial to the first and last 10 data points.

All of the algorithms were implemented using the NumPy library in Python.

## Sec. IV. Computational Results

### i. Training and Test Errors

The training errors and test errors for fitting a line, parabola, and 19th degree polynomial to the dataset are shown below:

#### Training errors:

Line: 2.242749386808539

Parabola: 2.125539348277377

19th degree polynomial: 0.02835144302630829

#### Test errors:

Line: 3.5278140684148744

Parabola: 9.13895508870405

19th degree polynomial: 30023572038.45924

It can be observed that the 19th degree polynomial has the lowest training error, which indicates that it fits the training data very well. However, the test error for the 19th degree polynomial is significantly higher than the test errors for the line and parabola. This suggests that the 19th degree polynomial overfits the training data and performs poorly on unseen data.

### ii. Testing with Subset of Data

Next, we tested the line, parabola, and 19th degree polynomial on a subset of the data consisting of the first 10 and last 10 data points. The test errors are shown below:

Line: 2.948751607976003

Parabola: 2.9353026962885598

19th degree polynomial: 81.9285731445318

It can be observed that the test errors for the line and parabola are very similar, and slightly higher than the training errors for these models. The test error for the 19th degree polynomial is significantly higher than the test errors for the line and parabola, indicating that it overfits the data even more severely when trained on a subset of the data.

Overall, these results demonstrate the importance of choosing an appropriate model complexity that balances the tradeoff between model fit and generalization to unseen data. The 19th degree polynomial performs very well on the training data, but poorly on the test data, highlighting the danger of overfitting.

## Sec. V. Summary and Conclusions

In conclusion, polynomial regression can be a powerful method for predicting data points, but the choice of polynomial degree depends on the specific dataset and application. In this implementation, we showed that the linear and quadratic models perform well on certain subsets of the data, while the 19th degree polynomial model has low training error but poor test performance. Further exploration and experimentation is needed to determine the optimal degree of polynomial for a given dataset.
