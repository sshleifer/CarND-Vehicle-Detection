import cv2
import numpy as np
import pickle

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/825 # meters per pixel in x dimension

M, Minv, mtx, dist = pickle.load(open('./pickled_data/camera_calibration.p', 'rb'))


def rgb_read(path):
    srcBGR = cv2.imread(path)
    return cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)


def undistort(image, mtx=mtx, dist=dist):
    return cv2.undistort(image, mtx, dist, None, mtx)


def get_s_channel(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    return hls[:,:,2]


def warp(img):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)
    return warped


def get_curvature(a, b, y):
    numer = (1 + (2*a*y+b)**2)**1.5
    denom = np.abs(2*a)
    return numer / denom

def get_x_for_max_y(bug, left_fit):
    return predict_poly(left_fit, bug.shape[0])


def get_offset(img, left_fit, right_fit):
    left_lane_pos_pixels = get_x_for_max_y(img, left_fit)
    right_lane_pos_pixels = get_x_for_max_y(img, right_fit)
    vehicle_pos_pixels = img.shape[1] / 2.  # vehicle in center of image
    center_of_lane_pos_px = np.mean([left_lane_pos_pixels, right_lane_pos_pixels])
    vehicle_offset_pixels = vehicle_pos_pixels - center_of_lane_pos_px
    vehicle_offset_meters = vehicle_offset_pixels * xm_per_pix
    return vehicle_offset_meters

def predict_poly(coeffs, val):
    return coeffs[0] * val**2 + coeffs[1] * val + coeffs[2]


def fit_poly(binary_warped, xm_per_pix=xm_per_pix, ym_per_pix=ym_per_pix):

    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                      (win_xleft_high,win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                      (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    y_eval = max(ploty) * ym_per_pix
    # Calculate the new radii of curvature
    left_curverad = get_curvature(left_fit_cr[0], left_fit_cr[1], y_eval)
    right_curverad = get_curvature(right_fit_cr[0], right_fit_cr[1], y_eval)
    offset = get_offset(out_img, left_fit, right_fit)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return (left_fit, right_fit,
            left_curverad, right_curverad, offset,
            left_lane_inds, right_lane_inds,
            out_img,  nonzerox, nonzeroy)


def draw_line(img, left_fit, right_fit, Minv):
    '''Draw the lane lines on the image `img` using the poly `left_fit` and `right_fit`.'''
    yMax = img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Calculate points.
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    # from Tips and Tricks Drawing submodule
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)


