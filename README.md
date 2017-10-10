**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_features.png
[image1b]: ./examples/not_car_features.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/test4.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I called `single_image_feature` from the lecture, for each vehicle and non vehicle image, using the below parameters.
The code sets `Xmat` in the notebook.



I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`), with the goal of improving my classifier. Out of the box, accuracy was 97.7% but by switching to `YCrCb` and increasing hist_bins=64, I was able to increase accuracy to 99.4%
```{python}
PARAMS = {
  'cell_per_block': 2,
  'color_space': 'YCrCb',
  'hist_bins': 32,
  'hog_channel': 'ALL',
  'orient': 9,
  'pix_per_cell': 8,
  'spatial_size': (16, 16)
 }
 ```

The following example uses `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![alt text][image1]
![alt text][image1b]


####2. Explain how you settled on your final choice of HOG parameters.
Took combo with best test accuracy after googling around a bit. The biggest jump was changing hist_bins to 36.


####3. Describe how (and identify where in your code) you trained a classifier 
`clf.fit` in the jupyter notebook uses the given data and my augmentations, which are `Xmat.pkl` and `y.pkl`.
I added about 200 hand labeled images. I also sneakily retrain on the full dataset after fitting on just the train data.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search. 

I used the lecture code for sliding windows but added a few more windows to search, 
basically so that there was lots of overlap and bigger windows at the top of the frame.


![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
Most performance optimization was by changing the arguments to `single_image_feature`.
Below is the heatmap for an example the classifier does well on.

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./P5_final.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I just followed the lecture code here. Record positive detections -> create heatmap -> require at least two windows have the object classified in them.
Then I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. 
I did not track blobs frame to frame because this seemed to hurt my performance at the beginning so I gave up on it.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took was to optimize the classifier by changing the features, and then once I got stuck, around 99.2%,
to try to add examples from the beginning of the video to the dataset.

I found it hard to calculate the hog features once per image, because I kept getting arrays of the wrong shape.
As a result, iterating on the video pipeline took a long time. I probably should have buckled down and got this part working earlier.
The full pipeline took 25 mins to run by the end!

My pipeline will probably not do well if it is not in the left lane (given the x_start parameter). 
I would like to make the classifier more robust by starting with something that is trained on imagenet or PASCAL VOC and not using Hog features.

I also tried only making sure a bounding box had to be in the same window two frames in a row (commented out in `video.py`),
but that didnt seem to help!

