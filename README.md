# Final Project Proposal
## MM Applications in Satellite Imagery
### Population Expansion and Coastal Reduction

This project is a collaborative effort of image processing and 
predictions for land use change.  Satellite imaging is convert to
data base format for building forecasting models of changes in patterns.  Two change patterns will be examined:  image indication of population growth (i.e. increase in light intensity) and coastal sea level rise impact (i.e. vegetation intensity).
The models will allow a user to select a future year to receive
feedback on the expected changes in land use availability based upon growth and/or coast reduction.  

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
5/ The model loss was 0.005

The python file that contains this code is "tensor flow predicion greenland.py"<br><br>
Run this file and you will see a predicted image






