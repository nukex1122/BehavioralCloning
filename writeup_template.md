# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/model_1.png "Model Architecture"
[image6]: ./examples/images1.jpg "Normal Image"
[image7]: ./examples/image2.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is built after modifications done to the one NVIDIA made.
It first crops the image so that only relevant part of the data is inputted to the Convolution Network, discarding extra data such as trees, mountains, etc. It crops the upper 65 pixels and the lower 20 pixels, outputting a shape of (75,320,3)

After that it passes a convolution network having 3 filters of filter size (5,5) and stride 2 followed by 2 filters of filter size (3,3) and stride 1. I tested with other larger and deeper models, but most overfitted on the data and weren't providing good results. 

The model includes ELU layers to introduce nonlinearity after each layer. The data is normalized in the model using a Keras lambda layer before it enters the convolution network. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (train_py.py line 96).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the sample data and augmented it to produce more data.
The issue with the given training data was that most of the images had a steering angle of 0.0 and the car was learning to incentivize that more. So when I was generating the samples, I sampled the data such that I only get 20% of the training data which had a steering angle of 0.0. This helped my model a lot. 
I also augmented the data by flipping every image and including the left and right cameras for training as well. 
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a conv net which didn't overfit the data and generalized well/ 

My first step was to use a convolution neural network model similar to the AlexNet/LeNet I thought this model might be appropriate because it captures a lot of information about the data, but the model was overfitting. 
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 


To combat the overfitting, I modified the model. 
After googling around, I came across NVIDIA's architecture and the model generalized well on it. I also augmented the data and fixed the imbalance as explained in the previous section.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (train_model.py lines 86-101) consisted of a convolution neural network with the following layers and layer sizes. 

![alt text][image5]

#### 3. Creation of the Training Set & Training Process

I used the sample training set, as my laptop doesn't have a gpu and I myself wasn't able to drive using the simulator. 

To augment the data sat, I also flipped images and angles thinking that this would increase the data available and would be helpful in generalizing the image. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by fixing the class imbalance upto some extent by removing about 80% of the images which had a steering angle of 0.0. I also included the images of the left camera and right camera. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
