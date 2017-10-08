def search_windows(img, windows, clf, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32, orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    '''make features and predict on each window'''
    # 1) Create an empty list to receive positive detection windows
    img_tosearch = img
    ctrans_tosearch = convert_color(img_tosearch, color_space)
    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        s1 = slice(window[0][1], window[1][1])
        s2 = slice(window[0][0], window[1][0])
        # 3) Extract the test window from original image
        hog_feat1 = hog1[s1, s2].ravel()
        hog_feat2 = hog2[s1, s2].ravel()
        hog_feat3 = hog3[s1, s2].ravel()
        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
        subimg = cv2.resize(img[s1, s2], (64, 64))
        spatial_features = bin_spatial(subimg, size=spatial_size)
        hist_features = color_hist(subimg, nbins=hist_bins)
        print(hog_features.shape, hog1.shape, subimg.shape)
        #
        # # Scale features and make a prediction
        test_features = np.concatenate((spatial_features, hist_features, hog_features))
        prediction = clf.predict(test_features)
        # 4) Extract features for that window using single_img_features()
        # features = single_img_features(test_img, color_space=color_space,
        #                                spatial_size=spatial_size, hist_bins=hist_bins,
        #                                orient=orient, pix_per_cell=pix_per_cell,
        #                                cell_per_block=cell_per_block,
        #                                hog_channel=hog_channel)
        # print(features.shape)
        # 6) Predict using your classifier

        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows