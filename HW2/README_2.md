# EE399 HW #2
Author: Mathew Garcia-Medina

## Abstract:

This project explores the use of eigendecomposition and singular value decomposition (SVD) for analyzing image data. Specifically, we use these techniques to analyze a matrix of grayscale images of faces from the Yale Face Database. We compute the correlation matrix between pairs of images, and use this to identify highly correlated and uncorrelated images. We then perform eigendecomposition and SVD on the image matrix to identify the principal component directions and the percentage of variance captured by each component. Finally, we compare the results of these two methods and discuss their strengths and weaknesses for analyzing image data.

## Sec. I. Introduction and Overview

Image data is used in many fields, including computer vision, machine learning, and medical imaging. Eigendecomposition and singular value decomposition (SVD) are powerful techniques for analyzing image data, as they allow us to identify the most important features of the images and reduce their dimensionality for further analysis. In this project, we apply these techniques to a matrix of grayscale images of faces from the Yale Face Database. We first compute the correlation matrix between pairs of images, and use this to identify highly correlated and uncorrelated images. We then perform eigendecomposition and SVD on the image matrix to identify the principal component directions and the percentage of variance captured by each component. Finally, we compare the results of these two methods and discuss their strengths and weaknesses for analyzing image data.

## Sec. II. Theoretical Background

Eigendecomposition and singular value decomposition are both methods for analyzing matrices. Eigendecomposition is used to decompose a matrix into a set of eigenvectors and eigenvalues, which can be used to identify the most important features of the data. Singular value decomposition, on the other hand, decomposes a matrix into a product of three matrices, which can be used to identify the principal component directions of the data. Both methods are commonly used for reducing the dimensionality of data and for identifying patterns in high-dimensional data.

## Sec. III. Algorithm Implementation and Development

We begin by importing the necessary libraries and loading the image data:

<img width="298" alt="image" src="https://user-images.githubusercontent.com/122642082/232923765-82de7f47-0232-4c73-b308-86f9717f5003.png">

Next, we compute the correlation matrix between the first 100 images in the matrix X and plot using the 'pcolor' function:

<img width="360" alt="image" src="https://user-images.githubusercontent.com/122642082/232925437-e4fe7bf6-0bea-4055-80de-b0f8bf6a9238.png">

From the correlation matrix, we identify the two most highly correlated images and the two most uncorrelated images and plot the resulting images:

<img width="612" alt="image" src="https://user-images.githubusercontent.com/122642082/232925852-b7c97fdc-f677-4401-99bb-1bff72372df8.png">

We then compute the using the 10x10 correlational matrix using the given data and the 'np.dot' function:

<img width="491" alt="image" src="https://user-images.githubusercontent.com/122642082/232928373-a17a4ba4-25f5-472a-b87d-b33c03f5dff3.png">

We then compute the matrix Y = XX^T, find the first six eigenvectors with the largest magnitude eigenvalue, use the 'eig' function to compute the eigenvectors and eigenvalues, and sort them using the 'np.argsort' function:

<img width="758" alt="image" src="https://user-images.githubusercontent.com/122642082/232928763-535300eb-e7a4-4b61-bf7a-12062158a5d8.png">

Then, to compute the first six principal component directions using SVD of X, we used the svd function from the NumPy library:

<img width="571" alt="image" src="https://user-images.githubusercontent.com/122642082/232929181-01999045-b2b5-4275-b228-55a30075957e.png">

Then to compute the normalized difference between the absolute value of the vectors v1 from (d) with the first SVD mode u1 from (e), we use the 'np.linalg.norm' function:

<img width="525" alt="image" src="https://user-images.githubusercontent.com/122642082/232937984-3d320238-71f4-4c20-9f08-d5a4500dee0b.png">

Finally, to compute the percentage of variance captured by each of the first 6 SVD modes, we use the S return value that contains vectors with singular values:

<img width="587" alt="image" src="https://user-images.githubusercontent.com/122642082/232938056-10c854f0-8eb0-476e-acc9-1da726aeb588.png">

## Sec. IV. Computational Results

In this section, we present the computational results of the different analyses performed on the Yale Faces dataset.

### a) 100x100 Correlation Matrix

We computed the 100x100 correlation matrix between the first 100 images in the matrix X. The correlation matrix was plotted using the pcolor function from the matplotlib library. The resulting plot is shown in Figure 1.

#### Figure 1: 100x100 Correlation Matrix

<img width="604" alt="image" src="https://user-images.githubusercontent.com/122642082/232925572-fe481bc0-84ab-4553-ab5f-371c4e2295ec.png">

From the correlation matrix, we can observe that the diagonal elements are equal to 1, which is expected since the correlation between an image and itself is perfect. We can also see that some pairs of images are highly correlated, while others are not.

### b) Most Correlated and Uncorrelated Images

From the correlation matrix in part (a), we determined the most highly correlated and most uncorrelated images. The most highly correlated images were images 89 and 89, while the most uncorrelated images were images 65 and 65. The resulting plot is shown in Figure 2.

#### Figure 2: Most Correlated and Uncorrelated Images

<img width="460" alt="image" src="https://user-images.githubusercontent.com/122642082/232925888-124ec9b6-4bdc-4271-86c5-e3600917f0cc.png">

### c) 10x10 Correlation Matrix

We also computed the 10x10 correlation matrix between images 1, 313, 512, 5, 2400, 113, 1024, 87, 314, and 2005. Below is the graph of the correlation matrix.

#### Figure 3: 10x10 Correlation Matrix

<img width="487" alt="image" src="https://user-images.githubusercontent.com/122642082/232935558-d7332e80-59be-4360-980f-1f59a6863b42.png">

### d & e) Eigenvectors and SVD Modes

We also computed the first six eigenvectors with the largest magnitude eigenvalues of the matrix Y=XX^T and the first six principal component directions (SVD modes) of the matrix X using SVD. The eigenvectors and SVD modes are shown in Figure 4 and Figure 5, respectively.

#### Figure 4: First 6 Eigenvectors with Largest Magnitude Eigenvalues

<img width="548" alt="image" src="https://user-images.githubusercontent.com/122642082/232935845-34ee0ae1-c6a5-4d72-8f09-4acee2d9fe7c.png">

#### Figure 5: First 6 Principal Component Directions

<img width="575" alt="image" src="https://user-images.githubusercontent.com/122642082/232936056-caff3f00-52e8-48c6-880b-cde07fbf3817.png">

### f) Difference Between v1 and u1

Below is the computed value for the difference between the two vectors.

#### Figure 6: Computed Value for Difference Between v1 & u1

<img width="587" alt="image" src="https://user-images.githubusercontent.com/122642082/232938652-42447c76-59dd-43e8-bcaa-e2a9d856a60a.png">

### g) Percentage of Variance for First 6 SVD Modes

Below are the plots of the first 6 SVD modes and their corresponding variance percentage.

#### Figure 7: First 6 SVD Modes & Corresponding Variance Percentage

<img width="847" alt="image" src="https://user-images.githubusercontent.com/122642082/232938715-b6ad2921-e71e-40a3-a4f1-2cd9755ae0e9.png">

## Sec. V. Summary and Conclusions

In this project, we applied PCA to analyze a dataset of 2414 images of faces. We computed the correlation matrix for a subset of the data and visualized it using pcolor. We also computed the first six eigenvectors with the largest magnitude eigenvalues using the matrix Y = XX^T, and the first six principal component directions using SVD of X.

We compared the first eigenvector obtained using PCA with the first SVD mode and found that they were very similar, with a small difference in their absolute values. Furthermore, we calculated the percentage of variance captured by each of the first six SVD modes, finding that the first mode captures about 23% of the variance, while the other modes capture much smaller percentages.

Finally, we plotted the first six SVD modes and observed that they correspond to different types of variations in the images, such as lighting changes and different facial expressions. We also noted that the SVD approach is computationally more efficient than the eigendecomposition approach for large datasets.

In conclusion, PCA is a powerful technique for analyzing large datasets and extracting the most important features. The application of PCA to the analysis of face images demonstrates its usefulness in the field of computer vision and pattern recognition.
