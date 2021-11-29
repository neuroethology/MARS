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



def get_angle(Ax, Ay, Bx, By):
    # adjusted with rotation 90
    # start and end point inversed, tail as origin A
    angle = (mh.atan2(Ax - Bx, Ay - By) + mh.pi/2.) % (mh.pi*2)
    return angle


def fit_ellipse(X,Y):
    data = [X.tolist(),Y.tolist()]
    mu = np.mean(data,axis=1)
    covariance = np.cov(data)
    rad = (-2 * np.log(1 - .75)) ** .5
    _,D,R = np.linalg.svd(covariance)
    normstd  = np.sqrt(D)

    # phi = mh.acos(R[0,0])
    # if phi < 0.:
    #    phi = (2.* mh.pi + phi)
    # phi %= (mh.pi*2.)
    # if Y[0] >= Y[-1]:
    #     phi += mh.pi

    a = rad * normstd[0]
    b = rad * normstd[1]

    cx = mu[0]
    cy = mu[1]

    phi  = (mh.atan2(np.mean(X[4:7]) - np.mean(X[:3]), np.mean(Y[4:7]) - np.mean(Y[:3])) + mh.pi / 2.) % (mh.pi * 2)

    theta_grid = np.linspace(-mh.pi,mh.pi,200)
    xs = cx + a * np.cos(theta_grid) * np.cos(phi) + b * np.sin(theta_grid) * np.sin(phi)
    ys = cy - a * np.cos(theta_grid) * np.sin(phi) + b * np.sin(theta_grid) * np.cos(phi)

    # draw ori
    ori_vec_v = np.array([mh.cos(phi),-mh.sin(phi)]) * a
    ori_vec_h = np.array([mh.sin(phi), mh.cos(phi)]) * b

    ell = {'cx': cx, 'cy': cy, 'ra': a, 'rb': b, 'phi': phi,
           'xs': xs, 'ys': ys, 'ori_vec_v': ori_vec_v, 'ori_vec_h': ori_vec_h}

    return ell


def bb_intersection_over_union(boxA,boxB,im_w,im_h):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    if xB > xA and yB > yA:
        interArea = (xB*im_w - xA*im_w + 1) * (yB*im_h - yA*im_h + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2]*im_w - boxA[0]*im_w + 1) * (boxA[3]*im_h - boxA[1]*im_h + 1)
        boxBArea = (boxB[2]*im_w - boxB[0]*im_w + 1) * (boxB[3]*im_h - boxB[1]*im_h + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0
    # return the intersection over union value
    return iou

def interior_angle(p0, p1, p2):
    def unit_vector(v):
        return v/np.linalg.norm(v)

    v0 = np.array(p0)-np.array(p1)
    v1 = np.array(p2)-np.array(p1)

    return mh.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))


def soc_angle(lam, x1, y1, x2, y2):
    x_dif = lam['xy']['centroid_x'](x2, y2) - lam['xy']['centroid_x'](x1, y1)
    y_dif = lam['xy']['centroid_y'](x2, y2) - lam['xy']['centroid_y'](x1, y1)
    theta = (np.arctan2(y_dif, x_dif) + 2 * np.pi) % 2 * np.pi
    ang = np.mod(theta - lam['xy_ang']['ori_body'](x1, y1), 2 * np.pi)
    return np.minimum(ang, 2 * np.pi - ang)


def facing_angle(lam, x1, y1, x2, y2):
    ell1 = (fit_ellipse(x1, y1))
    vec_rot = np.vstack((np.cos(lam['ell_ang']['phi'](ell1)), -np.sin(lam['ell_ang']['phi'](ell1))))
    c1 = np.vstack((lam['xy']['centroid_x'](x1, y1), lam['xy']['centroid_y'](x1, y1)))
    c2 = np.vstack((lam['xy']['centroid_x'](x2, y2), lam['xy']['centroid_y'](x2, y2)))
    vec_btw = c2 - c1
    norm_btw = np.linalg.norm(np.vstack((vec_btw[0, :], vec_btw[1, :])), axis=0)
    vec_btw = vec_btw / np.repeat([norm_btw], 2, axis=0)
    return np.arccos((vec_rot * vec_btw).sum(axis=0))


def angle_between(lam, x1, y1, x2, y2):
    ell1 = (fit_ellipse(x1, y1))
    ell2 = (fit_ellipse(x2, y2))
    vec_rot1 = np.vstack((np.cos(lam['ell_ang']['phi'](ell1)), -np.sin(lam['ell_ang']['phi'](ell1))))
    vec_rot2 = np.vstack((np.cos(lam['ell_ang']['phi'](ell2)), -np.sin(lam['ell_ang']['phi'](ell2))))
    return np.arccos((vec_rot1 * vec_rot2).sum(axis=0))


def dist_nose(lam, x1, y1, x2, y2):
    x_dif = lam['xy']['nose_x'](x2, y2) - lam['xy']['nose_x'](x1, y1)
    y_dif = lam['xy']['nose_y'](x2, y2) - lam['xy']['nose_y'](x1, y1)
    return np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)


def dist_centroid(lam, x1, y1, x2, y2):
    x_dif = lam['xy']['centroid_x'](x2, y2) - lam['xy']['centroid_x'](x1, y1)
    y_dif = lam['xy']['centroid_y'](x1, y1) - lam['xy']['centroid_y'](x2, y2)
    return np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)


def dist_body(lam, x1, y1, x2, y2):
    x_dif = lam['xy']['centroid_body_x'](x2, y2) - lam['xy']['centroid_body_x'](x1, y1)
    y_dif = lam['xy']['centroid_body_y'](x2, y2) - lam['xy']['centroid_body_y'](x1, y1)
    return np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)


def dist_head(lam, x1, y1, x2, y2):
    x_dif = lam['xy']['centroid_head_x'](x2, y2) - lam['xy']['centroid_head_x'](x1, y1)
    y_dif = lam['xy']['centroid_head_y'](x2, y2) - lam['xy']['centroid_head_y'](x1, y1)
    return np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)


def dist_head_body(lam, x1, y1, x2, y2):
    x1_dif = lam['xy']['centroid_head_x'](x1, y1) - lam['xy']['centroid_body_x'](x2, y2)
    y1_dif = lam['xy']['centroid_head_y'](x1, y1) - lam['xy']['centroid_body_y'](x2, y2)
    return np.linalg.norm(np.vstack((x1_dif, y1_dif)), axis=0)


def dist_gap(lam, x1, y1, x2, y2):
    # semiaxis length
    ell1 = (fit_ellipse(x1, y1))
    ell2 = (fit_ellipse(x2, y2))
    c_M_0 = np.multiply(lam['ell']['major_axis_len'](ell1), np.sin(lam['ell_ang']['phi'](ell1)))
    c_m_0 = np.multiply(lam['ell']['minor_axis_len'](ell1), np.cos(lam['ell_ang']['phi'](ell1)))
    c_M_1 = np.multiply(lam['ell']['major_axis_len'](ell2), np.sin(lam['ell_ang']['phi'](ell2)))
    c_m_1 = np.multiply(lam['ell']['minor_axis_len'](ell1), np.cos(lam['ell_ang']['phi'](ell2)))

    comb_norm = np.linalg.norm(np.vstack((c_M_0, c_m_0)), axis=0) + np.linalg.norm(np.vstack((c_M_1, c_m_1)), axis=0)
    return dist_body(lam, x1, y1, x2, y2) - comb_norm


def speed_head_hips(lam, xt1, yt1, xt2, yt2):
    dhead_x = lam['xy']['centroid_head_x'](xt2, yt2) - lam['xy']['centroid_head_x'](xt1, yt1)
    dhead_y = lam['xy']['centroid_head_y'](xt2, yt2) - lam['xy']['centroid_head_y'](xt1, yt1)
    dbody_x = lam['xy']['centroid_body_x'](xt2, yt2) - lam['xy']['centroid_body_x'](xt1, yt1)
    dbody_y = lam['xy']['centroid_body_y'](xt2, yt2) - lam['xy']['centroid_body_y'](xt1, yt1)
    return np.linalg.norm([np.vstack((dhead_x, dbody_x)), np.vstack((dhead_y, dbody_y))], axis=(0, 1))


def speed_centroid(lam, xt1, yt1, xt2, yt2):
    dx = lam['xy']['centroid_x'](xt2, yt2) - lam['xy']['centroid_x'](xt1, yt1)
    dy = lam['xy']['centroid_y'](xt2, yt2) - lam['xy']['centroid_y'](xt1, yt1)
    return np.linalg.norm([dx, dy], axis=0)


def speed_fwd(lam, xt1, yt1, xt2, yt2):
    cx1 = lam['xy']['centroid_x'](xt1, yt1)
    cy1 = lam['xy']['centroid_y'](xt1, yt1)
    cx2 = lam['xy']['centroid_x'](xt2, yt2)
    cy2 = lam['xy']['centroid_y'](xt2, yt2)
    dir_mot = get_angle(cx1, cy1, cx2, cy2)
    # slight change- original code used a 4-frame moving average to estimate dx and dy
    dx = cx2 - cx1
    dy = cy2 - cy1
    return np.multiply(np.linalg.norm([dx, dy], axis=0), np.cos(lam['xy_ang']['ori_body'](xt2, yt2) - dir_mot))


def radial_vel(lam, xt2, yt2, xt1, yt1, x2, y2):
    eps = np.spacing(1)
    # get the vector between the centroids of the two mice
    ddx1 = lam['xy']['centroid_x'](xt2, yt2) - lam['xy']['centroid_x'](x2, y2)
    ddy1 = lam['xy']['centroid_y'](xt2, yt2) - lam['xy']['centroid_y'](x2, y2)
    ddx1 = ddx1 / np.max((np.sqrt(ddx1 ** 2. + ddy1 ** 2.), eps))
    ddy1 = ddy1 / np.max((np.sqrt(ddx1 ** 2. + ddy1 ** 2.), eps))
    # calculate the velocity of the resident along that vector
    dx = lam['xy']['centroid_x'](xt2, yt2) - lam['xy']['centroid_x'](xt1, yt1)
    dy = lam['xy']['centroid_y'](xt2, yt2) - lam['xy']['centroid_y'](xt1, yt1)
    return dx * ddx1 + dy * ddy1


def tangential_vel(lam, xt2, yt2, xt1, yt1, x2, y2):
    eps = np.spacing(1)
    # get the vector orthogonal to the vector between the centroids of the two mice
    ddx1_T = -(lam['xy']['centroid_y'](xt2, yt2) - lam['xy']['centroid_y'](x2, y2))
    ddy1_T = -(lam['xy']['centroid_x'](xt2, yt2) - lam['xy']['centroid_x'](x2, y2))
    ddx1_T = ddx1_T / np.max((np.sqrt(ddx1_T ** 2. + ddy1_T ** 2.), eps))
    ddy1_T = ddy1_T / np.max((np.sqrt(ddx1_T ** 2. + ddy1_T ** 2.), eps))
    # calculate the velocity of the resident along that vector
    dx = lam['xy']['centroid_x'](xt2, yt2) - lam['xy']['centroid_x'](xt1, yt1)
    dy = lam['xy']['centroid_y'](xt2, yt2) - lam['xy']['centroid_y'](xt1, yt1)
    return dx * ddx1_T + dy * ddy1_T


def acceleration_head(lam, x2, y2, x1, y1, x0, y0):
    ax = lam['xy']['centroid_body_x'](x2, y2) - 2 * lam['xy']['centroid_body_x'](x1, y1) + \
         lam['xy']['centroid_body_x'](x0, y0)
    ay = lam['xy']['centroid_body_y'](x2, y2) - 2 * lam['xy']['centroid_body_y'](x1, y1) + \
         lam['xy']['centroid_body_y'](x0, y0)
    return np.linalg.norm([ax, ay], axis=0)


def acceleration_body(lam, x2, y2, x1, y1, x0, y0):
    ax = lam['xy']['centroid_body_x'](x2, y2) - 2 * lam['xy']['centroid_body_x'](x1, y1) + \
         lam['xy']['centroid_body_x'](x0, y0)
    ay = lam['xy']['centroid_body_y'](x2, y2) - 2 * lam['xy']['centroid_body_y'](x1, y1) + \
         lam['xy']['centroid_body_y'](x0, y0)
    return np.linalg.norm([ax, ay], axis=0)


def acceleration_ctr(lam, x2, y2, x1, y1, x0, y0):
    ax = lam['xy']['centroid_x'](x2, y2) - 2 * lam['xy']['centroid_x'](x1, y1) + lam['xy']['centroid_x'](x0, y0)
    ay = lam['xy']['centroid_y'](x2, y2) - 2 * lam['xy']['centroid_y'](x1, y1) + lam['xy']['centroid_y'](x0, y0)
    return np.linalg.norm([ax, ay], axis=0)


def crop_image(img, x, y, radius):
    im_h = img.shape[0]
    im_w = img.shape[1]

    pad_top = max(-min(int(y - radius), 0), 0)
    pad_left = max(-min(int(x - radius), 0), 0)
    pad_bottom = max(max(int(y + radius), im_h) - im_h, 0)
    pad_right = max(max(int(x + radius), im_w) - im_w, 0)

    xr, yr = np.ix_(range(min(int(x - radius), 0), max(int(x - radius), im_w)),
                    range(min(int(y - radius), 0), max(int(y + radius), im_h)))
    xr = np.pad(xr, ((pad_top, pad_bottom), (0, 0)), 'reflect')
    yr = np.pad(yr, ((0, 0), (pad_left, pad_right)), 'reflect')
    return img[yr, xr]


def pixel_change_local(lam, img1, img0, x1, y1, x0, y0, l):
    radius = l / 20.  # l is the length of a mouse
    patch1 = crop_image(img1, x1, y1, radius)
    patch0 = crop_image(img0, x0, y0, radius)
    return (np.sum((patch1 - patch0) ** 2)) / float((np.sum((patch0) ** 2)))


def pixel_change_ubbox(lam, bb1, bb2, bb10, bb20, img1, img0):
    # this is a very dumb feature, you shouldn't use it
    f1_bb = lam['bb']['overlap_bboxes'](bb1, bb2)
    f2_bb = lam['bb']['overlap_bboxes'](bb10, bb20)

    if f1_bb > 0. or f2_bb > 0.:
        xmin11, xmax11 = bb10[[0, 2]]
        ymin11, ymax11 = bb10[[1, 3]]
        xmin12, xmax12 = bb20[[0, 2]]
        ymin12, ymax12 = bb20[[1, 3]]

        xmin21, xmax21 = bb1[[0, 2]]
        ymin21, ymax21 = bb1[[1, 3]]
        xmin22, xmax22 = bb2[[0, 2]]
        ymin22, ymax22 = bb2[[1, 3]]

        tmp1 = img1[int(min(ymin11, ymin12, ymin21, ymin22)):int(max(ymax11, ymax12, ymax21, ymax22)),
               int(min(xmin11, xmin12, xmin21, xmin22)):int(max(xmax11, xmax12, xmin21, xmin22))]
        tmp2 = img0[int(min(ymin11, ymin12, ymin21, ymin22)):int(max(ymax11, ymax12, ymax21, ymax22)),
               int(min(xmin11, xmin12, xmin21, xmin22)):int(max(xmax11, xmax12, xmin21, xmin22))]

        return (np.sum((tmp2 - tmp1) ** 2)) / float((np.sum((tmp1) ** 2)))
    else:
        return 0


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
