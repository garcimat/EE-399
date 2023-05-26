# EE399 HW #6

Title: Analysis of SHRED Model Performance on SST Data: A Study on Time Lag and Sensor Variability

Author: Mathew Garcia-Medina

## Abstract:

This report presents an analysis of the performance of the SHRED (SHallow REcurrent Decoder) model applied to sea-surface temperature (SST) data. The study focuses on two factors that may impact the model's performance: time lag and sensor variability. The SHRED model combines a recurrent layer (LSTM) with a shallow decoder network (SDN) to reconstruct high-dimensional spatio-temporal fields from a trajectory of sensor measurements.

In this analysis, we investigate the impact of different time lags on the SHRED model's performance. Additionally, we explore how the number of sensors used in the model affects its reconstruction accuracy. The NOAA Optimum Interpolation SST V2 dataset is employed for the experiments, and the performance is evaluated using mean squared error (MSE) as the metric.

The report presents the theoretical background of the SHRED model and the algorithm implementation. It describes the process of preprocessing the data, generating input sequences, and creating the training, validation, and test datasets. The training procedure and hyperparameters of the SHRED model are also discussed.

The computational results section presents the findings of the analysis. Performance as a function of time lag and sensor variability is examined, and the corresponding mean squared errors are reported. The results provide insights into the optimal choice of time lag and the impact of sensor variability on the model's performance.

In summary, this report provides a comprehensive analysis of the SHRED model's performance on SST data, considering the influence of time lag and sensor variability. The findings contribute to a better understanding of the model's capabilities and can guide the selection of suitable parameters for future applications in spatio-temporal data reconstruction.

## Sec. I. Introduction and Overview

The field of spatio-temporal data analysis plays a crucial role in various domains, including climate science, environmental monitoring, and predictive modeling. The ability to reconstruct high-dimensional spatio-temporal fields from a trajectory of sensor measurements is essential for understanding complex dynamics and making accurate predictions. The SHRED (SHallow REcurrent Decoder) model offers a promising approach to address this challenge.

The objective of this report is to analyze and evaluate the performance of the SHRED model applied to sea-surface temperature (SST) data. The SST data provides valuable insights into the Earth's climate system and has applications in climate research, marine biology, and weather forecasting. By investigating the performance of the SHRED model on SST data, we aim to gain insights into its capabilities and limitations.

The report is structured as follows. Section II provides a theoretical background on the SHRED model, explaining its architecture and the underlying principles of recurrent neural networks and decoder networks. This section establishes the foundation for understanding the model's operation and its potential for spatio-temporal data reconstruction.

Section III details the implementation and development of the SHRED model. It covers the preprocessing of the SST data, including data loading and scaling using the MinMaxScaler. The process of generating input sequences for the model and creating the training, validation, and test datasets is described. Additionally, the hyperparameters and training procedure of the SHRED model are discussed.

In Section IV, the computational results of the analysis are presented. The performance of the SHRED model is evaluated by measuring the mean squared error (MSE) between the reconstructed data and the ground truth. Two aspects of model performance are investigated: the influence of the time lag parameter and the impact of sensor variability. The results provide insights into the optimal choice of time lag and the sensitivity of the model to the number of sensors.

Finally, Section V summarizes the findings and draws conclusions based on the analysis. It discusses the implications of the results, highlights the strengths and limitations of the SHRED model, and suggests potential areas for further research and improvement.

Overall, this report aims to contribute to the understanding of the SHRED model's performance on SST data and its applicability in spatio-temporal data reconstruction. The insights gained from this analysis can aid researchers and practitioners in making informed decisions when utilizing the SHRED model for analyzing and predicting spatio-temporal phenomena.

## Sec. II. Theoretical Background

The SHRED (SHallow REcurrent Decoder) model is a network architecture designed for reconstructing high-dimensional spatio-temporal fields from a trajectory of sensor measurements. This section provides a theoretical background on the SHRED model, introducing the key components and underlying principles.

The SHRED architecture combines a recurrent layer, specifically the Long Short-Term Memory (LSTM) network, with a shallow decoder network (SDN). The LSTM network is well-suited for modeling sequential data and capturing temporal dependencies. It consists of memory cells and gates that control the flow of information, allowing the network to retain long-term dependencies.

The SDN serves as the decoder part of the SHRED model, responsible for reconstructing the high-dimensional spatio-temporal fields. It is a feed-forward network composed of fully connected layers. The SDN takes the output of the LSTM network as input and transforms it to generate the reconstructed field.

To apply the SHRED model to SST data, the input sequences are generated by selecting a trajectory of sensor measurements over a given time lag. The sensor locations are randomly chosen, and the number of sensors can be varied to analyze its impact on the model's performance.

The training of the SHRED model involves optimizing the network parameters to minimize a suitable loss function, typically mean squared error (MSE), between the reconstructed field and the ground truth. The training is performed using a training dataset, and the model's performance is evaluated on validation and test datasets.

The SHRED model offers the advantage of combining the temporal modeling capabilities of the LSTM network with the flexibility and expressiveness of the SDN. This architecture enables the reconstruction of complex spatio-temporal fields from incomplete and noisy sensor measurements.

In summary, the SHRED model provides a powerful framework for spatio-temporal data reconstruction. By leveraging the LSTM network and SDN, it can capture temporal dependencies and generate accurate reconstructions of high-dimensional fields. The next section will delve into the implementation and development of the SHRED model for SST data analysis.

## Sec. III. Algorithm Implementation and Development

This section presents the implementation and development of the SHRED model for analyzing sea-surface temperature (SST) data. It covers the data preprocessing, generation of input sequences, creation of training and validation datasets, and the training procedure of the SHRED model. Code snippets are provided to illustrate the key steps in the algorithm.

Data Preprocessing
The first step in the algorithm is to preprocess the SST data. This involves loading the data and applying data scaling using the MinMaxScaler from the sklearn library. The following code snippet demonstrates the data preprocessing steps:

<img width="374" alt="image" src="https://github.com/garcimat/EE-399/assets/122642082/07297689-fe9b-4f7f-b844-219cae6b5496">
<img width="374" alt="image" src="https://github.com/garcimat/EE-399/assets/122642082/b48222b6-0c05-48d8-b8ea-5c48067df746">

Generating Input Sequences
The next step is to generate input sequences for the SHRED model. These sequences consist of a trajectory of sensor measurements over a specified time lag. The sensor locations are randomly selected, and the input sequences are stored in the all_data_in array. The following code snippet demonstrates this process:

<img width="374" alt="image" src="https://github.com/garcimat/EE-399/assets/122642082/75e6c10b-2c0b-47c5-bef7-9d9da80263a3">
<img width="553" alt="image" src="https://github.com/garcimat/EE-399/assets/122642082/9416feda-c607-4dcc-a3a0-33b71facb03d">

Creating Training, Validation, and Test Datasets
To train and evaluate the SHRED model, we need to divide the data into training, validation, and test sets. This is done by selecting indices based on the specified time lag. The following code snippet demonstrates the process:

<img width="589" alt="image" src="https://github.com/garcimat/EE-399/assets/122642082/ff6c7919-fbbc-4096-8b0c-785847aa5a83">
<img width="872" alt="image" src="https://github.com/garcimat/EE-399/assets/122642082/937e3f30-a8f2-4f03-8676-7cd4a6056873">

Training the SHRED Model
The final step is to train the SHRED model using the training and validation datasets. The model architecture and hyperparameters are defined, and the fit function is called to initiate the training process. The following code snippet demonstrates this step:

<img width="1032" alt="image" src="https://github.com/garcimat/EE-399/assets/122642082/8eb176fb-b387-4c93-ae75-fcc2fee608f6">

This code snippet creates an instance of the SHRED model with the specified architecture and hyperparameters. The fit function trains the model using the training dataset, validates the performance on the validation dataset, and performs early stopping based on the specified patience parameter.

The implementation and development of the SHRED model for SST data analysis is a multi-step process that involves data preprocessing, input sequence generation, dataset creation, and model training. The code snippets provided illustrate the key steps in the algorithm. The next section will present the computational results obtained from applying the SHRED model to SST data.

## Sec. IV. Computational Results

In this section, we present the computational results obtained from applying the SHRED model to analyze sea-surface temperature (SST) data. We evaluate the performance of the model in terms of reconstruction accuracy and forecasting capabilities. Code snippets are provided to showcase the analysis and visualization of the results.

Performance Analysis as a Function of Time Lag
To assess the impact of the time lag variable on the performance of the SHRED model, we vary the lag length and analyze the resulting reconstruction accuracy. The following code snippet demonstrates how to conduct this analysis:

<img width="642" alt="image" src="https://github.com/garcimat/EE-399/assets/122642082/06c7ed61-f59e-48d9-b9ff-1983a3d59316">

By varying the lag_lengths and performing the necessary computations, this code snippet calculates the mean squared error (MSE) for each lag length. The MSE reflects the reconstruction accuracy of the SHRED model, with lower values indicating better performance. The resulting MSE scores can be further analyzed or visualized to understand the relationship between time lag and model performance.

Performance Analysis with Gaussian Noise
To assess the performance of the SHRED model under different noise levels, we can introduce Gaussian noise to the input data and evaluate the model's reconstruction accuracy. The following code snippet demonstrates this analysis:

<img width="486" alt="image" src="https://github.com/garcimat/EE-399/assets/122642082/ab2e8f95-220c-4d6e-bd90-cedeb831909f">

By varying the noise_levels and introducing Gaussian noise to the input data, this code snippet evaluates the model's reconstruction accuracy under different noise levels. The resulting MSE scores provide insights into the model's robustness to noise.

Performance Analysis as a Function of the Number of Sensors
To analyze the impact of the number of sensors on the SHRED model's performance, we can vary the number of sensor locations used for input sequence generation. The following code snippet demonstrates this analysis:

<img width="486" alt="image" src="https://github.com/garcimat/EE-399/assets/122642082/b0ae37c4-c21b-45bc-9f24-57aa55264e76">

By varying the sensor_counts and generating input sequences with different numbers of sensors, this code snippet evaluates the model's performance. The resulting MSE scores provide insights into how the number of sensors affects the reconstruction accuracy.

The analysis presented in this section demonstrates the performance of the SHRED model under various conditions, including different time lags, noise levels, and numbers of sensors. The code snippets provided illustrate how to conduct the analyses and compute the corresponding MSE scores. The next section summarizes the findings and presents the conclusions derived from the computational results obtained.

## Sec. V. Summary and Conclusions

In this study, we applied the SHRED (SHallow REcurrent Decoder) model to analyze sea-surface temperature (SST) data. The SHRED model combines a recurrent layer (LSTM) with a shallow decoder network (SDN) to reconstruct high-dimensional spatio-temporal fields from a trajectory of sensor measurements. The aim of this research was to evaluate the performance of the SHRED model in terms of reconstruction accuracy and forecasting capabilities.

The computational results revealed valuable insights into the performance of the SHRED model under different scenarios. We conducted analyses to assess the impact of the time lag variable, noise levels, and the number of sensors on the model's performance. The mean squared error (MSE) was used as a metric to evaluate the reconstruction accuracy, with lower values indicating better performance.

Regarding the performance analysis as a function of the time lag variable, we observed that the choice of lag length influenced the reconstruction accuracy of the SHRED model. Varying the lag length allowed us to identify an optimal value that resulted in improved reconstruction accuracy.

The analysis of performance with Gaussian noise demonstrated the model's robustness to varying levels of noise. We introduced Gaussian noise to the input data and evaluated the model's ability to reconstruct the original signal. The results provided insights into the model's performance under noisy conditions.

Furthermore, the analysis of performance as a function of the number of sensors shed light on the relationship between the number of sensors and reconstruction accuracy. Varying the number of sensor locations used for input sequence generation allowed us to understand the impact of sensor density on the model's performance.

Overall, the SHRED model exhibited promising performance in reconstructing spatio-temporal fields from sensor measurements. The findings suggest that the SHRED model has the potential to be a valuable tool for analyzing and forecasting complex spatio-temporal data such as SST.

In conclusion, this study demonstrated the effectiveness of the SHRED model in analyzing SST data. The model's ability to reconstruct high-dimensional spatio-temporal fields from sensor measurements was evaluated under various conditions. The results provide valuable insights for understanding the factors that influence the model's performance and offer potential directions for further research and improvements. The SHRED model holds promise for applications in climate science, environmental monitoring, and other domains requiring accurate spatio-temporal data analysis.
