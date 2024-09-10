# Final Project Part B: Deep Neutral TensorFlow's Keras Model

## Overview
The following source files are provided in this folder:

Data_Cleaning_Glacier_Retreat : This Jupyter Notebook provides the code to
prepare the initial data files for modelling.  The original data files are
included in the "resources" folder.

Data Models 1 through 4:  Each Jupyter notebook contains one Tensorflow Keras
sequential model based upon a specific set of data.  The data for each model is
included in the "resources" folder.
             
## Part 1: Summary of data preparation

The data sets including primarily floating point numbers, except the output data
was in integer format. The output or "y" column was already in a "0-1" format,
which did not require any additional processing.  The data was then split into a
training and testing dataset with train_test_split.  StandardScaler was used to
scale the float point numbers that made up the "X" data set.  

## Part 2:  Compile and evaluate a deep neural network model.
A deep neural network was created with no less than two hidden layers and an
 output layer are included in each model.  Each model was iterated for the following
 properties:

A Keras deep learning module was set up with the following parameters:
    1.  Test Data Size:  This was adjusted because the number of rows was small
        compared to ideal deep neural network datasets.
    2.  Layer Inputs:  These were varied based upon input dimensions
    3.  Epochs:  Number of epoches were adjusted between 50 and 500.
Each model was compiled and fit using the binary crossentropy losss function, the
adam optimizer, and the accuracy evaluation metric.  

### Part 3: Predicting Glacier Retreat 
The final presented model for each set of parameters performed as follows:

Model 1 Climate Change Parameters
Loss: 0.6750372052192688, Accuracy: 0.6428571343421936

Model 2 Population, Farm, and Finance Parameters
Loss: 0.7428156137466431, Accuracy: 0.5555555820465088

Model 3 Change in Temperature Parameters
Loss: 0.8554542064666748, Accuracy: 0.7777777910232544

Model 3 was also run with a Keras Tuner hyperparameter input.
This lead to a better first pass than the other models but still was
limited in performance. 

Keras Tuner Predicted:
Loss: 0.5031033158302307, Accuracy: 0.9090909361839294

Keras Tuner Actual First Pass:
Loss: 1.162386417388916, Accuracy: 0.6666666865348816

The main difference with the Keras Tuner parameters was identifying 4 hidden
layers and proposed layer inputs, whereas manually we were only using 2 hidden
layers and few different input, iterated 2 to 3 times.
hidden_nodes_layer1 = 11
hidden_nodes_layer2 = 6
hidden_nodes_layer3 = 16
hidden_nodes_layer4 = 11

Model 4 Deforestation Parameters
Loss: 91.33601379394531, Accuracy: 0.6000000238418579


### Part 4: Recommendations
The glacier change data is actually limited to a few decades.  Although the
glaciers are monitored for melting because of established sea rise, predicting
changes in future scenarios seems very challenging.  The glacier positions 
sometimes increased and sometimes decreased year to year, and the parameters
studied against the glacier retreat (or lack of) did not seem to be a good fit
to what was actually happening.  It is possible that the glacier response lags 
behind the other parameters, or the glacier response is impacted by parameters
not included, such as snowfall in the arctic region.  

Future modeling work includes:
 1. Finding larger sources of data, possibly on the monthly, weekly, or even daily
    level that could contribute to more data faster.

 2. Shifting the results in time, lead or lag, in case the responses are different
    in time.

 3. Including parameters such as snowfall in the analysis.

 4. Using other models such as linear regression and Prophet with glacier area
    change, instead of classification models for "retreat" and "no retreat"

 5. Using Keras Tuner as the preliminary model for each set of parameters.

