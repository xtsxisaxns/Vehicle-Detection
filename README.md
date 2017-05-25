# Vehicle-Detection-Project-Carnd

## Introduction
This project comes from udacity self-driving-car nanodegree.The goal is develop a software pipeline to detect vehicles in 
image and video.The steps of the detection are:
* Perform a Histogram of Oriententad Gradient(HOG) feature extraction on a labeled training set of images and train a Linear 
SVM classifier.
* To improve accuracy,apply a color transform and append binned color features,as well as histograms of color,to HOG feature 
vector.
* Note:for the first steps,normalize the features because these features are not in the same scale.And randomize a selection 
for training and testing.
* Implement a sliding-window technique and use trained classifier to detect vehicles in images.
* Run detection pipeline on video.There must be a lot wrong detection bboxes.So create a heat map of recurring detections frame 
by frame to reject outliers and follow detected vehicles.

## Training dataset 
Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

## Histogram of Oriented Gradients (HOG)
Apply hog feature extraction to the training dataset.here we use hog extraction method in skimage.[Here is skimage hog link](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html).The hog images of car and non-car are:
![](https://github.com/nicholas-tien/Vehicle-Detection-Project-Carnd/blob/master/examples/hog.png?raw=true)


## Train a linear SVM classifier
For a better result,training images are converted to HLS color space.All three channels of a image are used.Besides,bined-color feature and histograms of color are appened.Here we use LinearSVC classifer from sklearn.[Here is the sklearn svc link](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

## Sliding window search
We use four sliding windows of different size.Sliding window first slides from left side to right side  of the image.Then slides from top to down.The overlap percentage of adjacent Windows is 0.75.By sliding window strategy,we can get a lot of 
windows.Get the roi image of each window positon,resize the roi image to (64,64,3).Use the tarined LinearSVC classifer to 
classify it.After this,we'll get some windows that may have a car in it.A detection result is:
![](https://github.com/nicholas-tien/Vehicle-Detection-Project-Carnd/blob/master/examples/all_window.png?raw=true)
![](https://github.com/nicholas-tien/Vehicle-Detection-Project-Carnd/blob/master/examples/hot_window.png?raw=true)


## Filter wrong detections
From the above hot_windows images,we can see that there are some wrong detections.To filter them out,a heatmap method and threshold filter is used.In a single image,only apply threshold.In video,last n frames are took into account.When a new frame 
is processed.The oldest is passed out.In video processing,a threshold of n is used.Because wrong detections rarely appear all the time,this can filter some wrong detections.Some process images as follows:
![](https://github.com/nicholas-tien/Vehicle-Detection-Project-Carnd/blob/master/examples/heatmap.png?raw=true)
![](https://github.com/nicholas-tien/Vehicle-Detection-Project-Carnd/blob/master/examples/heatmap_threshod.png?raw=true)
![](https://github.com/nicholas-tien/Vehicle-Detection-Project-Carnd/blob/master/examples/detection.png?raw=true)












