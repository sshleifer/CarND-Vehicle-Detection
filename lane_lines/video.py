import cv2
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
import matplotlib.pyplot as plt

from lane_lines.line import Lane
from lane_lines.fit_poly import (fit_poly, warp, Minv, draw_line, undistort,
                                 predict_poly, xm_per_pix)
from lane_lines.sobel_utils import get_threshd_image


def plot_gray(img, cmap='gray'):
    fig, axes = plt.subplots(ncols=1, figsize=(10, 10))
    axes.imshow(img, cmap=cmap)
    axes.axis('off')
    return axes


def write_text(output, txt, height):
    cv2.putText(output, txt, (10, height),
                cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 250, 0), 10)


def show_diagnostic(big_tuple, img):
    (left_fit, right_fit,
            left_curverad, right_curverad, offset,
            left_lane_inds, right_lane_inds,
            out_img,  nonzerox, nonzeroy) = big_tuple
    return out_img


def is_valid_lane_width(img, left_fit, right_fit):
    width_pixels = []
    for y_frac in [.5, .75, 1.]:
        y = img.shape[0] * y_frac
        width = predict_poly(right_fit, y) - predict_poly(left_fit, y)
        width_meters = width * xm_per_pix
        width_pixels.append(width_meters)
        if width_meters < 3.5 or width_meters > 5:
            print(y_frac, width_meters)
            return False, width_pixels

    return True, width_pixels


def is_valid_curve(last_l, last_r, new_l, new_r):
    chg = max(np.abs(last_l - new_l), np.abs(last_r - new_r))
    lr_curvature_gap = np.abs(new_l - new_r)
    if ((chg > 10000) or (lr_curvature_gap > 10000) or
            (new_l > 1e5) or (new_r > 1e5)):
        return False
    else:
        return True


def annotate_output(output, lcurves, rcurves, offsets, overwrite, widths):
    'write curvature and offset to top of image'
    write_text(output, 'Left curvature: {:.0f} m'.format(np.mean(lcurves[-5:])),
               50)
    write_text(output, 'Right curvature: {:.0f} m'.format(np.mean(rcurves[-5:])),
               100)
    write_text(output, 'Vehicle is {:.2f}M right of center'.format(np.mean(offsets[-5:])),
               150)
    write_text(output, 'Using latest fit = {}, min_width = {:.1f}'.format(overwrite, min(widths)),
               200)
    return output


def video_pipeline(clip):


    left_lane = Lane(curve=3000)
    right_lane = Lane(curve=3000)
    l_curve = 3000
    r_curve = 3000
    logs = []
    imgs = []
    diag = []
    offsets = []
    lcurves= []
    rcurves = []

    for i, img in enumerate(clip.iter_frames()):
        udist = undistort(img)
        binary_warped = warp(get_threshd_image(udist))
        big_tuple = fit_poly(binary_warped)
        left_fit, right_fit, left_curve, right_curve, offset = big_tuple[:5]
        logs.append([
            left_fit, right_fit, left_curve, right_curve, offset])

        overwrite = is_valid_curve(l_curve,
                                   r_curve,
                                   left_curve,
                                   right_curve)
        valid_width, widths = is_valid_lane_width(udist, left_fit, right_fit)
        overwrite = overwrite and valid_width
        if overwrite or i == 0:
            l_curve = left_curve
            r_curve = right_curve
            left_fit = left_fit
            right_fit = right_fit
            left_lane = Lane(left_fit)
            right_lane = Lane(right_fit, right_curve)
            lcurves.append(l_curve)
            rcurves.append(r_curve)
            offsets.append(offset)

        diag.append(show_diagnostic(big_tuple, udist))
        output = draw_line(udist, left_lane.fit, right_lane.fit, Minv)
        output = annotate_output(output, lcurves, rcurves, offsets, overwrite,
                                 widths)
        imgs.append(output)

    return imgs, diag