import numpy as np


def make_many_windows(image):
    windows = slide_window(image, x_start_stop=[500, None], y_start_stop=[400, 560],
                           xy_window=(64, 64), xy_overlap=(0.0, 0.0))

    y_start_stop = [400, 680]  # Min and max in y to search in slide_window()
    x_start_stop = [500, None]  # Min and max in y to search in slide_window()

    windows2 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                            xy_window=(92, 74), xy_overlap=(0.25, 0.25))

    windows3 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                            xy_window=(120, 84), xy_overlap=(0.5, 0.5),)

    windows4 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                            xy_window=(148, 94), xy_overlap=(0.5, 0.5))

    windows5 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                            xy_window=(176, 104), xy_overlap=(0.5, 0.5))

    windows6 = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                            xy_window=(204, 114), xy_overlap=(0.5, 0.5))
    return windows + windows2 + windows3 + windows4 + windows5 + windows6


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list
