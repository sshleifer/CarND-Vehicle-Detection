
from detection.constants import PARAMS
from detection.lesson_functions import draw_boxes
from detection.heatmap import get_heatmap, draw_labeled_bboxes, get_bboxes
from detection.window import slide_window, make_many_windows
from detection.search_classify import search_windows

from scipy.ndimage.measurements import label

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
    heatmaps = []
    for i, image in tqdm(enumerate(clip.iter_frames())):
        windows = make_many_windows(image)
        hot_windows = search_windows(image, windows, clf, **search_params)
        heatmap = get_heatmap(hot_windows, image, threshold=threshold)
        heatmaps.append(heatmap)

        labels = label(heatmap)
        bboxes = get_bboxes(labels)
        keep = []
        if len(last_hot_windows) < 5:
            keep = bboxes
        else:
            for w in last_hot_windows[-5:]:
                for cand in bboxes:
                    # print(cand, w)
                    if cand in w:
                        keep.append(cand)

        #last_hot_windows.
        # window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)
        output = draw_boxes(image, keep)
        last_hot_windows.append(keep)
        imgs.append(output)

    return imgs, heatmaps


def draw_heat(image, heatmap, debug=False):
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    if not debug:
        return draw_img

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()