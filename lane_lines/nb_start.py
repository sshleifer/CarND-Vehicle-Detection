from __future__ import division
# @nolint
from collections import defaultdict
import pandas as pd
from pandas.util.testing import assert_frame_equal
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random, permutation, randn, normal, uniform, choice

np.set_printoptions(precision=3)
from pandas.tools.plotting import autocorrelation_plot, lag_plot, scatter_matrix
import os
import funcy
import json
import re
from scipy import stats
# import seaborn as sns
# sns.set_context('notebook')
# sns.set_style('whitegrid')
import datetime
import glob
import time
from tqdm import tqdm
# import pysftp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import tensorflow as tf
from PIL import Image
# plt.rc('figure', figsize=(10, 6))

# Handy scikit-learn stuff

from sklearn.cluster import KMeans, AffinityPropagation, MiniBatchKMeans
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import (recall_score, precision_score, precision_recall_curve,
                             classification_report, precision_recall_fscore_support)
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import normalize, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (LinearRegression, LogisticRegression, LassoLarsIC,
                                  Lasso, ElasticNet, SGDClassifier, RidgeCV, LassoCV)
from sklearn import clone, linear_model
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import RandomForestClassifier

# convenience units
k = 1e3
mm = 1e6
bb = 1e9
