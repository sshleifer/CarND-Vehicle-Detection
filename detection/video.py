
from detection.constants import PARAMS
from detection.heatmap import get_heatmap, draw_heat
from detection.window import slide_window, make_many_windows
from detection.search_classify import search_windows
from lane_lines.nb_start import *
from tqdm import tqdm


def process_image(image, clf,  threshold=1, search_params=PARAMS):
    windows = make_many_windows(image)#slide_window(image, y_start_stop=[ystart, ystop])

    hot_windows = search_windows(image, windows, clf, **search_params)
    heatmap = get_heatmap(hot_windows, image, threshold=threshold)
    # window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)
    return draw_heat(image, heatmap)



def video_pipeline(clip, clf, threshold=1, search_params=PARAMS):


    last_hot_windows = []
    imgs = []
    for i, image in tqdm(enumerate(clip.iter_frames())):
        windows = make_many_windows(image)
        hot_windows = search_windows(image, windows, clf, **search_params)
        heatmap = get_heatmap(hot_windows, image, threshold=threshold)
        keep = []
        if len(last_hot_windows) < 5:
            keep = heatmap
        else:
            for w in last_hot_windows[-5:]:
                for cand in heatmap:
                    print(cand, w)
                    if cand in w:
                        keep.append(cand)

        #last_hot_windows.
        # window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)
        output = draw_heat(image, keep)
        last_hot_windows.append(keep)
        imgs.append(output)

    return imgs#, diag