# Flower Recognition

The model classify the image of the input flower in one of the 5 categories (daisy, rose, sunflower, tulip and dandelion). Especially, this system is helpful for recognizing pictures that have fastened form and structure of flowers. Then we have a tendency to create use neural network that processes the individual pixels of a picture and then training of model and prediction.

## How to run?
    Run the model.py and a my_model will be created having trained model. Then run the run.py and enjoy.

## Programming Approach:
    •	CNN Network
    •	ReLU threshold function in input layer
    •	Softmax threshold function in output layer
    •	Adam optimizer
    •	4 Convolutional and 4 Pooling Layers 
    •	512 hidden neurons and 5 output neurons


## Programming Tools
    The following programming tools are used in this project:
    •	Python
    •	Tensorflow
    •	Keras
    •	Numpy
    •	Matplotlib
    •	PyQt5 for GUI
    
## Model Design:
    The CNN model id designed in following steps:
    • The images for Training and Testing are separated in 80%:20% respectively.
    • The Images are given a rescale of 1/255 and then loaded in (224, 244) pixels.
    • We have used 4 convolutional layers and 4 pooling layers with different parametric values.
    • A dropout of 0.5 is given to model.
    • A flatten layer is added
    • A hidden layer with 512 number of neurons is added
    • And an output layer with 5 neurons is added
    • Then the Adam optimizer and Relu & Softmax threshold functions are used.

## Dataset
    Dataset is taken from the following link which is a Kaggle account. 
   `https://www.kaggle.com/alxmamaev/flowers-recognition`

## Model Accuracy
    We have got an accuracy of 73% for final prediction and the accuracy on which model is trained is 91%.
