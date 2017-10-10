import cv2
import numpy as np

from lane_lines.fit_poly import get_s_channel


def apply_threshold(img, thresh_min, thresh_max):
    '''return ((img > thresh_min) & (img < thresh_max)).astype(int)'''
    return ((img > thresh_min) & (img < thresh_max)).astype(int)


def mag_thresh(gray, sobel_kernel=3, thresh_min=30, thresh_max=100):
    '''gray should be grayscaled'''
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    return apply_threshold(gradmag, thresh_min, thresh_max)


def abs_sobel_thresh(gray, orient='x', sobel_kernel=3,  thresh_min=10, thresh_max=160):
    '''Apply x or y gradient with the OpenCV Sobel() function and take the absolute value'''
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    return apply_threshold(scaled_sobel, thresh_min, thresh_max)


def direction_thresh(gray, sobel_kernel=3, thresh_min=.8, thresh_max=1.2):
    ''' Take the absolute value of the gradient direction'''
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    return apply_threshold(absgraddir, thresh_min, thresh_max)


def get_threshd_image(img, x_min=10, y_min=160, x_max=160, y_max=160):
    # TODO(SS): also use R channel
    schannel = get_s_channel(img)
    x_sobel_threshd = abs_sobel_thresh(schannel, orient='x', thresh_min=x_min, thresh_max=x_max)
    y_sobel_threshd = abs_sobel_thresh(schannel, orient='y', thresh_min=y_min, thresh_max=y_max)
    combined_mask = x_sobel_threshd | y_sobel_threshd #| mag_threshd # dont use dir_bthresh
    return combined_mask.astype('uint8')