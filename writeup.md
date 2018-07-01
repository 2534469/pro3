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

[image1]: ./examples/histogram.png "Angle distribution"
[image2]: ./examples/central_image.png "Central"
[image3]: ./examples/central_cropped.png "Cropped Image"
[image4]: ./examples/central_resized.png "Resized Image"
[image5]: ./examples/central_noized.png "Noised Image"
[image6]: ./examples/flipped.png "Flipped Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

As a base for my model I took the Nvidia convolutional network, with the input of (160, 320, 3) sized images.
I cropped the initial image sizes removing the upper part and the lower part with the car nose and downsized the images having the output of size(32, 160, 3), see lines 116-117 of model.py or my jupyter notebook. First I tried to greyscale images, but there was no gain of it, so I removed greyscaling. As in Nvida CNN I have 5 Convolutional layers, following by a Flatten layer and 5 Dense layers (see model.py line 130-136).


#### 2. Attempts to reduce overfitting in the model

The model contains 5 dropout layers in order to reduce overfitting (model.py lines 123, 125, 133). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 

![alt text][image1]

As you can see from distribution of angles values and since the car goes long straight lane without turns, the values around 0 are the most common in the dataset.
In order to prevent the car to go always straightforward I removed all the data with the absolute angle value less than abs_steering_min=0.1 (line 18).

#### 3. Model parameter tuning

The model used an adam optimizer, but I also specified the learning rate of 0.01 (see line 138).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, though it was difficult to record only recovering. I added also different scenes for recovering for the same turn, i.e. part of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the Nvidia CNN.

My first step was to use this convolution neural network because of their success described in the paper. I thought this model might be appropriate because Nvidia engineers use it for similar purpose and train their model also on the gathered track pictures.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I resized the images and added dropouts so the model can generalize.

Then I decreased the learnning rate and encreased the epochs. I also added incrementally data for different drives, including also in the opposite direction. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, when there was no right line. To improve the driving behavior in these cases, I recorded many drives passing that corner, I also converted images from RGB to YUV and changed drive.py accordingly.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 121-141) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 YUV image   						
| Cropping image 70 from up and 25 from lower end     	|   65x320x3	|
| Resizing     	|   30x160x3	|
| Convolution layer 1 24 x 5x5 kernel 	| 2x2 stride 	|
| Convolution layer 2 36 x 5x5 kernel 	| 2x2 stride 	|
| Dropout      	| 0.2 probability of dropping		|
| Convolution layer 3 48 x 5x5 kernel 	| 2x2 stride 	|
| Dropout      	| 0.2 probability of dropping		|
| Convolution layer 4 64 x 1x1 kernel 	| 1x1 stride 	|
| Dropout      	| 0.2 probability of dropping	|
| Convolution layer 5 64 x 1x1 kernel 	| 1x1 stride 	|
| Flatten, Dense     	| 1164	neurons|
| RELU					|	relu of flatten layer|
| Dropout      	| 0.4 probability of dropping	|
| Dense     	| 100 neurons	|
| RELU					|	relu of dense layer|
| Dropout      	| 0.4 probability of dropping	|
| Dense     	| 50 neurons	|
| RELU					|	relu of dense layer|
| Dense     	| 10 neurons	|
| RELU					|	relu of dense layer|
| Dense     	| 1 value	|
| Hyperbolic tangent activation 					|	activation of dense layer|
| Loss function				| Mean squared error        									|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I cropped the image to rwmove sky, trees and other non-relevant information that would only confuse the network:

![alt text][image3]

After that I resized the image to half of the initial length and height:

![alt text][image4]


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the situation approaching some road edge. 

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would encrease the data for training. For example, here is an image that has then been flipped:

![alt text][image6]

In order to augment data and get even more I also added noise to pictures with the absolute angle greater than 0.2:

![alt text][image4]

After the collection process, I had 23071 number of data lines. The number of images created was at least 6x times more. I had to take at least 5 drives and at least 10 drives of the cureves.  


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the validation loss not decreasing.
