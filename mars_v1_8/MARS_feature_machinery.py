from __future__ import division, print_function
import numpy as np
from itertools import islice
import multiprocessing as mp
import math as mh
from util.seqIo import *
import scipy
import scipy.io as sio
import cmath as chm
import dill
import progressbar
np.seterr(divide='ignore', invalid='ignore')


flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))



def syncTopFront(f,num_frames, num_framesf):
    return int(round(f / (num_framesf - 1) * (num_frames - 1))) if num_framesf > num_frames else \
        int(round(f / (num_frames - 1) * (num_framesf - 1)))


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def expanding_window(seq, final_n):
    """ Returns an expanding window over data until it reaches a certain width"""
    "   s -> (s0), (s0,s1), (s0,s1,s2), ... "
    it = iter(seq[:final_n])
    result = tuple(islice(it, 1))
    if len(result) == 1:
        yield result
    for elem in it:
        result = result + (elem,)
        yield result


def column_iterator(array, windows):
    num_columns = np.shape(array)[1]
    for i in range(num_columns):
        yield [array[:, i], windows]


def compute_JAABA_feats(starter_features, windows=[]):
    """This function computes the JAABA windowed features for a
    Inputs:
      starter_features: The features being transformed by windowed functions.
        Shape: [numFrames]x[numFeatures]
        Type: np-array

      windows: A list of window sizes to use to compute these features.
        Shape: [Undef]
        Type: Integer-list
        Note: all window size must be 0

    Outputs:
      window_features: The new, windowed version of the features.
        Shape: [numFrames]x[(numFeatures)*(numFunctions)*(numWindows)]
        Type: np-array
        Note: numFunctions is define in 'get_JAABA_feats'
    """

    if not windows:
        return starter_features
    else:
        total_feat_num = np.shape(starter_features)[1]

        bar = progressbar.ProgressBar(widgets=['WinFeats ', progressbar.Percentage(), ' -- ',
                                               progressbar.FormatLabel('Feature %(value)d'), '/',
                                               progressbar.FormatLabel('%(max)d'), ' [', progressbar.Timer(), '] ',
                                               progressbar.Bar(), ' (', progressbar.ETA(), ') '], maxval=total_feat_num)
        pool = mp.Pool()
        window_features = np.concatenate(
            list(bar(pool.imap(compute_win_feat_wrapper2, column_iterator(starter_features, windows)))), axis=1)
        pool.close()
        pool.join()
        return window_features


def compute_win_feat_wrapper2(starter_featurePLUSwindows):
    starter_feature, windows = starter_featurePLUSwindows
    window_feature = compute_win_feat2(starter_feature, windows)
    return window_feature


def compute_win_feat2(starter_feature, windows = [3, 11, 21]):
    """This function computes the window features from a given starter feature.
    Inputs:
      starter_feature: The feature being transformed by windowed functions.
        Shape: [numFrames]x1
        Type: np-array

      windows: A list of window sizes to use to compute these features.
        Shape: [Undef]
        Type: Integer-list
        Note: all window size must be 0


    Outputs:
      window_feature_matrix: a matrix of features
        Shape: [numWindowFeatures]x[numFrames]
        Type: np-array
    """
    # Define the functions being computed.
    fxns = [np.min, np.max, np.mean, np.std]
    num_fxns = len(fxns)

    # Get the number of frames
    number_of_frames = np.shape(starter_feature)[0]

    # Count the number of windows.
    num_windows = len(windows)

    # Number of additional features should simply be (the number of windows)*(the number of fxns) --in our case, 12.
    num_feats = num_windows*num_fxns

    # Create a placeholder for the features.
    features = np.zeros((number_of_frames,num_feats))

    # Loop over the window sizes
    for window_num,w in enumerate(windows):
            # Iterate with a given window size.
        # Get the space where we should put the newly computed features.
        left_endpt = window_num*num_fxns
        right_endpt = window_num*num_fxns + num_fxns

        # Compute the features and store them.
        features[:, left_endpt:right_endpt] = get_JAABA_feats2(starter_feature=starter_feature, window_size=w)

    return features


def get_JAABA_feats(starter_feature, window_size=1):
    """ Function thats computes the 4 JAABA features for a given window size. (min,max,std.dev.,mean)"""
    # Get the number of frames.
    number_of_frames = np.shape(starter_feature)[0]
    # Get the radius of the window.
    radius = (window_size - 1)/2

    # Set the functions.
    fxns = [np.min, np.max, np.mean, np.std]
    num_fxns = len(fxns)

    # Make a placeholder for the window features we're computing.
    window_feats = np.zeros((number_of_frames, num_fxns))
    win_mat = []
    # Looping over each frame, compute the functions on this window.
    for frame_num in range(number_of_frames):
        left_endpt, right_endpt = get_window_endpts(current_time=frame_num, total_length=number_of_frames, radius=int(radius))

        for fxn_num in range(num_fxns):
            fxn = fxns[fxn_num]
            win_mat.append(starter_feature[left_endpt:right_endpt])
            window_feats[frame_num,fxn_num] = fxn(starter_feature[left_endpt:right_endpt])

    return window_feats


def get_JAABA_feats2(starter_feature, window_size=1):
    # Get the number of frames.
    number_of_frames = np.shape(starter_feature)[0]
    # Get the radius of the window.
    radius = (window_size - 1)/2
    radius = int(radius)
    r = int(radius)
    row_placeholder = np.zeros(window_size)
    column_placeholder = np.zeros(number_of_frames)

    row_placeholder[:r] = np.flip(starter_feature[1:(radius+1)],0)
    row_placeholder[r:] = starter_feature[:(radius+1)]

    column_placeholder[:-radius] = starter_feature[radius:]
    column_placeholder[-radius:] = np.flip(starter_feature[-(radius+1):-1],0)

    # Create the matrix that we're going to compute on.
    window_matrix = scipy.linalg.toeplitz(column_placeholder, row_placeholder)
    
    # Set the functions.
    fxns = [np.min, np.max, np.mean, np.std]
    num_fxns = len(fxns)

    # Make a placeholder for the window features we're computing.
    window_feats = np.zeros((number_of_frames, num_fxns))

    # Do the feature computation.
    for fxn_num, fxn in enumerate(fxns):
        if (window_size <= 3) & (fxn == np.mean):
            window_feats[:, fxn_num] = starter_feature
        else:
            window_feats[:, fxn_num] = fxn(window_matrix, axis=1)
    return window_feats


def get_window_endpts(current_time, total_length, radius):
    left_endpt = max(0, current_time - radius)
    right_endpt = min(total_length - 1, current_time + radius)

    return left_endpt, right_endpt


def normalize_pixel_data(data,view):
    if view == 'top':
        fd = [range(40, 49)]
    elif view == 'front':
        fd = [range(47,67)]
    elif view == 'top_pcf':
        fd = [range(40,57)]
    else:
        return data
    fd = list(flatten(fd))
    md = np.nanmedian(data[:, :, fd], 1, keepdims=True)
    data[:, :, fd] /= md
    return data


def clean_data(data):
    """Eliminate the NaN and Inf values by taking the last value that was neither."""
    idx = np.where(np.isnan(data) | np.isinf(data))
    if idx[0].size > 0:
        for j in range(len(idx[0])):
            if idx[0][j] == 0:
                data[idx[0][j], idx[1][j],idx[2][j]] = 0.
            else:
                data[idx[0][j], idx[1][j],idx[2][j]] = data[idx[0][j] - 1, idx[1][j],idx[2][j]]
    return data
