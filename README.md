# Final Project:  Artificial Intelligence and Machine Learning
## MM Applications in Satellite Imagery
### Glacier Reduction and Retreat with Climate Change Impacts

This project is a collaborative effort of image processing and 
predictions for land use change.  Satellite imaging is converted to
data base format for building forecasting models of changes in patterns.  
Two change patterns will be examined:  
    A.  Image indication of change and future image generation and coastal sea level rise impact (i.e. ice loss)
    B.  Categorial models using deep neural modeling for classification prediction (i.e. glacier retreat)
 

## Time Series Prediction of Satellite Imagery

The objective of this portion of the project is to use a neural network to predict what a satellite image will be in the 
future base on a set of satellite images of a period of time. This was accomplished as follows:
<br><br>
1/ Read in a set of 8 .JPG satellite images for Greenland spanning 8 days<br>
2/ Reduce the size of the images by 90% and keep the scale - I tried running the script with the size of the images as is and
the script consumed too much RAM, even for Google Colab<br>
3/ I created a 4 layer Time Distributed Neural Network using TensorFlow. I tried doing this with PyTorch but the RAM consumption
was too much<br>
4/ I ran the model with 10 epochs and generated a predicted image<br>
5/ The preicted image is "predicted image.jpg"
5/ The model loss was 0.005

The python file that contains this code is "tensor flow predicion greenland.py"<br><br>
Run this file and you will see a predicted image.

## TensorFlow’s Keras Model with Keras Tuner (Classification)

Part B utilized various climate change data collections in an attempt to predict 
arctic glacier retreat or advance on an annual basis.  The data was broken into 
four different sets and models, with the output for all models being the Arctic 
Glacier Retreat “Yes/No” classification.  Inputs include such parameters as global 
temperature change, sea rise levels, change in temperature by country, world population 
data, farming data, economic data, and rates of deforestion, separated into four d
ifferent datasets.
The folder / directory “Data_Analytics_Component   “ contains the jupyter notebook
files, associated saved models, outputs, and input data for the TensorFlow’s Keras Model.
Inside “Data_Analytics_Component”, a readme file details the model descriptions, 
model settings, and other processing information.  The information presented here i
s only an overview. 
The jupyter notebook “Data_Cleaning_Glacier_Retreat “ contains the python code 
for cleaning and preparing the data files for the models.  
The models are included in “Data_Model_1” through 4.  The keras tuner output for 
Model 3a is included in “Keras_Tuner_Model_3”
The resources folder includes the following:
1.	The data files prefix numbered “00-“ through “07-“ are the unprepared data files.  
2.	The data files suffix number *_parameters-1 through -4 are the prepared data sets for read into the model.  
3.	Various data characteristics for data discovery purposes.  

Generally, the models were not incredibly reliable, with typical output around 50% (on two categories).   One iteration of the Change in Temperature by Country reached over 70% with manual manipulation of hidden layer properties and test set size.  The selected model for Keras Tuner achieved over 60% on first pass following application of the Keras Tuner components. 

Recommendations for future work with the deep neural models are detailed in the Keras folder.  





