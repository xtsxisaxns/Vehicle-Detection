# Vehicle-Detection-Project-Carnd
##Introduction
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










