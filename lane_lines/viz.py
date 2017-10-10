import matplotlib.pyplot as plt

def plot_side_by_side(left_img,  right_img, left_title='original', right_title='transformed',
                      figsize=(20, 10), leftCmap=None, rightCmap=None):
    """Display`left_img` and `right_img` side by side with titles"""
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    if leftCmap == None:
        axes[0].imshow(left_img)
    else:
        axes[0].imshow(left_img, cmap=leftCmap)
    axes[0].set_title(left_title)

    if rightCmap == None:
        axes[1].imshow(right_img)
    else:
        axes[1].imshow(right_img, cmap=rightCmap)
    axes[1].set_title(right_title)


