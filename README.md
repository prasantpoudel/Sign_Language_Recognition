# Sign_Language_Recognition

Dataset from kaggle::  https://www.kaggle.com/datasets/datamunge/sign-language-mnist
First : Create and Enviroment of python latest vesrion (option)

use command : pip install -r requirements.txt
Then You can create own data set using data_Collection.py use S key to capture the images

Then train the model (You can change the Architecture of the keras model or make the custom model)
According to the label on the data, change  the output Dense layer number i have given 26 for english alphabets.

Then test the model either through test.py in which open cv was used that uses realtime image from webcam  
or if you have the image then test the model in the training.ipynb
