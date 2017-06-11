# **Traffic Sign Recognition** 

## Traffic sign recognition utilizing a LeNET convolutional neural network

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./SignChart.png "SignChart"
[image2]: ./ImageProcessing.JPG "ImageProcessing"
[image3]: ./WebSamples.png "WebSamples"
[image4]: ./VisualFeatures.png "VisualFeatures"
[image5]: ./VisualFeatures1.png "VisualFeatures1"
[image6]: ./VisualFeatures2.png "VisualFeatures2"
[image7]: ./VisualFeatures3.png "VisualFeatures3"
[image8]: ./VisualFeatures4.png "VisualFeatures4"

---

### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

Please consider this README.md as my writeup, and addtionally see the [project code here](https://github.com/nategreco/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) for further reference.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

Simple functions from the python standard and numpy libraries were all that was necessary.  Using the len() function I gathered the size of the set in number of samples, and using the numpy shape attribute for numpy arrays I returned the dimensions and channels of the image.  See results here:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is 32 x 32 pixels with 3 channels (BGR)
* The number of unique classes/labels in the data set is 43


#### 2. Include an exploratory visualization of the dataset.

The chart below shows indicates the number of samples per each sign classification type:
![Classification chart][image1]

Additionally, in the Jupyter Notebook I have plotted one of each image classification along with the descripton from the csv file as an example.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

There was a lot of experimentation in this step.  Initially, I attempted only to clean-up the image and highlight features.  The function 'process_image()' was used to both maximize contrast and sharpen the image.  Afterwards the image was normalize from -1.0 to 1.0 for each element.  The thinking was higher contrast and sharper image would distinguish each classification type further from each other.  This theory proved true and training accuracy increased faster and further, however, the prediction operation failed to recognize any of the web images.  This is because the web images introduced variations that did not exist in the training set such as skew, rotation, displacement, etc.  Therefore the 'augment_image' function was implemented for the training and validation sets to introduce these varations.  This immediately improved the detection of the web image samples.

I then experimented further with other processing with mixed results.  Other processing that was tested:
* BGR to HSV conversion
* Grayscale conversion
* Canny edges image added as a 4th channel to the BGR image

Ultimately, the BGR only input image was found to have the best results in final accuracy during training, test set accuracy, and web image accuracy.  As of right now the code can be toggled between all the input types by adjusting the 'n_channels' variable at step 0 and commenting in/out the HSV conversion.  Examples of the original, augmented, processed, and canny images are below:

![Examples][image2]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.).

My final model was the standard LeNet, similar to what was used in the Lenet lab, however I modified it to handle different inputs depending on channels and also added dropout after the first fully connected layer:

| Layer         		| Description		        										| 
|:---------------------:|:-----------------------------------------------------------------:| 
| Input         		| 32x32x3 BGR image (also tested HSV, Grayscale, and BGR + Canny) 	| 
| Convolution	     	| 1x1 stride, valid padding, outputs 28x28x6 						|
| RELU					|																	|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 						|
| Convolution		    | 1x1 stride, valid padding, outputs 10x10x16 						|
| RELU					|																	|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 						|
| Flatten				| Outputs 400x1 			      									|
| Fully connected 		| Outputs 120x1 			      									|
| RELU					|																	|
| Dropout				| 50% keep probability 												|
| Fully connected 		| Outputs 84x1 			      										|
| RELU					|																	|
| Fully connected 		| Outputs 43x1 			      										|
| Softmax 				| Outputs 43x1 			      										|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The training model utilized the Adam optimizer and batching with a batch size of 128.  Additionally, I implemented the exponential decay of the learning rate and a hyperparameter for dropout keep probability.  The dropout function helped prevent overfitting, meanwhile the exponential decay allowed me to start with a higher learning rate.  Starting a with a higher learning rate allowed me to make larger improvements in accuracy in the initial epochs, however, the decayed lower learning rate helped avoid over-shooting during back propagation and getting stuck in local minimums due too high of a learning rate.  Regarding the number of epochs, a larger number was chosen but a break condition was added when the target accuracy of 0.95 was reached.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

Much of the steps have been outlined above, but key ...

My final model results were:
* Validation set accuracy of 95% 
* Test set accuracy of 95.17%
* Web sample accuracy of 100.0%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report.

Here are five German traffic signs that I found on the web:

![Internet samples][image3]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image 							|     Prediction	        					| 
|:---------------------------------:|:---------------------------------------------:| 
| No entry     						| No entry  									| 
| Right-of-way at next intersection | Right-of-way at next intersection 			|
| Speed limit (60khm/h)				| Speed limit (60khm/h)							|
| Beware of ice/snow      			| Beware of ice/snow  				 			|
| Yield								| Yield			    							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

For the most part, the prediction probabilities make sense for the selected signs.  The first, second, and third likely signs usually bear a similar geometric shape and generally the same appearance.  In particular, every speed limit sign usually has the other speed limit signs as possibilities, this is due in part by the fact that they are all red circles with white spac in the middle and black text.  This makes sense particularly when you look at initial images and realize how granulated the text becomes and it's difficult for the training set to learn the characters due to pixelation.  However the speed limit signs still did well with the 60 km/hr sign 99.94% probable.

The signs that were the greatest struggle to distinguish from one another were the Right-of-way, Beware of ice/snow, and slippery road signs.  Again, this is becuase the basic geometry, a red triangle with white interior and black graphic, was the same.  For image 2 probability of right away was only 88.21%, and for image 4 was only 80.06% that it was beware of ice and snow.  Image 4 was actually the most difficult to identify and while experimenting with the training model this one was often detected as right-of-way or slippery road instead.  This is understandable when you look at the processed 32x32 image examples and see how pixelated in snowflake graphic is, at that resolution is distorted enough that it's difficult even for a human to distinguish between the others.  I believe a higher resolution input image correct this.

Image 1:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.9998%     			| No entry  									| 
| 0.00019%     			| Stop 											|
| 0.00001%				| No passing 									|
| 0.00000%	      		| Bumpy Road					 				|
| 0.00000%			    | Bicycles crossing    							|

Image 2:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 88.2027% 				| Right-of-way at the next intersection 		|
| 11.7153% 				| Beware of ice/snow 							|
| 0.08105% 				| Speed limit (100km/h) 						|
| 0.00041% 				| Vehicles over 3.5 metric tons prohibited 		|
| 0.00020% 				| Dangerous curve to the right 					|

Image 3:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.9412% 				| Speed limit (60km/h) 							|
| 0.05824% 				| Speed limit (50km/h) 							|
| 0.00045% 				| Speed limit (80km/h) 							|
| 0.00007% 				| Wild animals crossing 						|
| 0.00005% 				| No passing 									|

Image 4:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 80.0562% 				| Beware of ice/snow 							|
| 7.58810% 				| Road narrows on the right 					|
| 5.55834% 				| Slippery road 								|
| 3.21945% 				| Bicycles crossing 							|
| 1.31002% 				| Double curve 									|

Image 5:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.000% 				| Yield 										|
| 0.00000% 				| Priority road 								|
| 0.00000% 				| No passing 									|
| 0.00000% 				| No vehicles 									|
| 0.00000% 				| Speed limit (50km/h) 							|


### (Optional) Visualizing the Neural Network
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![Visual Features 1][image4]
![Visual Features 2][image5]
![Visual Features 3][image6]
![Visual Features 4][image7]
![Visual Features 5][image8]


