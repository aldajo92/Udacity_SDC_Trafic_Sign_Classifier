# **Udacity Self Driving Car: Trafic Sign Classifier** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The following explanation is asociated with the [Traffic_Sign_Classifier](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) notebook.

---

## Build a Traffic Sign Recognition Project ##

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/exploration.png "Exploration"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/amounts.png "Amounts"
[test_image1]: ./test_images/image3.png "Traffic Sign 1"
[test_image2]: ./test_images/image4.jpg "Traffic Sign 2"
[test_image3]: ./test_images/image5.png "Traffic Sign 3"
[test_image4]: ./test_images/image6.png "Traffic Sign 4"
[test_image5]: ./test_images/image7.png "Traffic Sign 5"

## Rubric Points
In this project, the following [rubric points](https://review.udacity.com/#!/rubrics/481/view) was considered individually and describe how each point in the implementation was addressed.  

---

## 1. Data Set Summary & Exploration

The pandas library was used to load the csv file and get the information for the traffic signs data set:

```
Number of training examples = 34799
Number of training validation = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

#### Exploratory visualization of the dataset.

In the exploratory visualizaiton of the dataset, the following capture was taken, where each image is associated with a class and a correspondent label.

![alt text][image1]

Then a bar chart was made showing the amount of examples asociated with each class available in the training set.

![alt text][image4]

## 2. Design and Test a Model Architecture

As first step, the data was shuffle using the following methods from `sklearn.utils`:

```python
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
```

After that, the images was normalized using the following function:
```
def normalize(data):
    return data / 255 * 0.8 + 0.1
```

The idea to normalize data is increase the accuracy, beacause was used before another approach converting examples to grayscale but this not result with a a consice prediction. The normalization will return a better result as will be show later.

#### Final Model Architechture

The LeNet model was implemented based from lesson 8 with a change to handle images with 3-channels instead of 1-channel and modified the number of labels from 10 to 43 `len(all_labels)`.


| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 1     	| 1x1 stride, VALID padding, output = 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 14x14x6   |
| Convolution 2  	    | 1x1 stride, VALID padding, output = 10x10x16  |
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 5x5x16    |
| Flatten				| output = 400									|
| Fully connected		| input = 400, output = 120       	            |
| RELU					|												|
| Fully connected		| input = 120, output = 84       	            |
| RELU					|												|
| Fully connected		| input = 84, output = 10       	            |

To train the model I used 20 epochs, a batch size of 128 and a learning rate of 0.001.

For the training optimizers, the `softmax_cross_entropy_with_logits` was used to get a tensor representing the mean loss value applied to tensorflow. `reduce_mean` to compute the mean of elements across dimensions of the result. Finally the minimize to the AdamOptimizer was applied on previous result.

The final model validation Accuracy was 0.960.
 

## 3. Test a Model on New Images

<!-- #### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify. -->
The traffic signs found on the web are the followings:

| Name | Image |
| :-----------------:|:------------------------: |
| image 1 |  ![][test_image1] |
| image 2 |  ![][test_image2] |
| image 3 |  ![][test_image3] |
| image 4 |  ![][test_image4] |
| image 5 |  ![][test_image5] |

Here are the results of the prediction:

| name | Image			        |     Prediction	        					|
|:---------------------:|:---------------------:|:---------------------------------------------:|
| image 1 | No entry       		| No entry   									|
| image 2 | Yield     			| Yield 										|
| image 3 | No entry				| No entry										|
| image 4 | No entry				| No entry										|
| image 5 | Stop      			| Stop     		    							|
| image 6 | Speed limit (70km/h)	| Speed limit (30km/h)							|
| image 7 | Keep Right			| Keep right									|

The model was able to correctly predict 6 other 7 traffic signs, which gives an accuracy of 86%.

The comparison between the accuracy with the testing sample (0.960) and the lower number of images for `Speed limit 70` in contrast with other speed limits images, the wrong prediction have a small quantity of examples for this label on the data sample. Adding variations for this images, suchs as inverting, rotating or augmenting might haved increased the accuracy.

The following output, is the result obtained in the notebook:
```
image1.png:
No entry: 100.00%
Speed limit (20km/h): 0.00%
Stop: 0.00%
Speed limit (30km/h): 0.00%
Yield: 0.00%

image2.png:
Yield: 100.00%
No passing for vehicles over 3.5 metric tons: 0.00%
Road work: 0.00%
No passing: 0.00%
Beware of ice/snow: 0.00%

image3.png:
No entry: 100.00%
Speed limit (20km/h): 0.00%
Speed limit (30km/h): 0.00%
Stop: 0.00%
No passing: 0.00%

image4.jpg:
No entry: 100.00%
Speed limit (20km/h): 0.00%
Speed limit (30km/h): 0.00%
Speed limit (50km/h): 0.00%
Speed limit (60km/h): 0.00%

image5.png:
Stop: 99.97%
Bicycles crossing: 0.02%
Wild animals crossing: 0.01%
Road work: 0.00%
Speed limit (30km/h): 0.00%

image6.png:
Speed limit (20km/h): 94.93%
Speed limit (30km/h): 5.07%
No entry: 0.00%
Speed limit (60km/h): 0.00%
Speed limit (70km/h): 0.00%

image7.png:
Keep right: 100.00%
Go straight or right: 0.00%
Go straight or left: 0.00%
General caution: 0.00%
End of no passing: 0.00%
```

## 4. Conclusions:
- Increasing the number of epocs and batch size is critical to get a good prediction, this values was changed based on the lesson 10 and 11.
- The amount of images affects the final result for the predictions. This conclusion is appreciated for the prediction of the test image for `Speed limit 70` label, where we see a wrong result considering the accuracy obtained by the model.
- For this implementation, we consider the dataset provided, but adding variations for the images used to train for each label, such as, images rotations, could improve the final result, but this not was part of this implementation.



