from __future__ import print_function,division
import numpy as np
import scipy.io as sp
import os
import sys
import cv2
from collections.abc import Iterable
import scipy.spatial.distance as dist
import scipy.signal as sig
from skimage.transform import resize as rs
import warnings
import csv
import json
import cmath as cmh
import scipy.io as sio
import numpy.core.records as npc
import joblib
import yaml
import copy
import progressbar
import pdb

warnings.filterwarnings('ignore')
sys.path.append('./')
from util.genericVideo import *
from MARS_feature_machinery import *
import MARS_output_format as mof
import MARS_feature_lambdas as mars_lambdas
import MARS_legacy_feature_extractors as mlf


flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, Iterable) else (a,)))

def load_pose(pose_fullpath):
    try:
        with open(pose_fullpath, 'r') as fp:
            pose = json.load(fp)
        return pose
    except Exception as e:
        raise e


def generate_valid_feature_list(cfg):
    num_mice = len(cfg['animal_names']) * cfg['num_obj']
    mice = ['m'+str(i) for i in range(num_mice)]
    # TODO: multi-camera support
    # TODO: replace hard-coded feature names with keypoints from the project config file.
    cameras = {'top':   ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base'],
               'front': ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base',
                         'left_front_paw', 'right_front_paw', 'left_rear_paw', 'right_rear_paw']}
    inferred_parts = ['centroid', 'centroid_head', 'centroid_body']

    lam = mars_lambdas.generate_lambdas()
    have_lambdas = [k for f in lam.keys() for k in lam[f].keys()]

    feats = {}
    for cam in cameras:
        pairmice = copy.deepcopy(mice)
        feats[cam] = {}
        for mouse in mice:
            feats[cam][mouse] = {}
            feats[cam][mouse]['absolute_orientation'] = ['phi', 'ori_head', 'ori_body']
            feats[cam][mouse]['joint_angle'] = ['angle_head_body_l', 'angle_head_body_r', 'angle_nose_neck_tail']
            feats[cam][mouse]['joint_angle_trig'] = ['sin_angle_head_body_l', 'cos_angle_head_body_l', 'sin_angle_head_body_r', 'cos_angle_head_body_r']
            feats[cam][mouse]['fit_ellipse'] = ['major_axis_len', 'minor_axis_len', 'axis_ratio', 'area_ellipse']
            feats[cam][mouse]['distance_to_walls'] = ['dist_edge_x', 'dist_edge_y', 'dist_edge']
            feats[cam][mouse]['speed'] = ['speed', 'speed_centroid', 'speed_fwd']
            feats[cam][mouse]['acceleration'] = ['acceleration_head', 'acceleration_body', 'acceleration_centroid']

        pairmice.remove(mouse)
        for mouse2 in pairmice:
            feats[cam][mouse+mouse2] = {}
            feats[cam][mouse+mouse2]['social_angle'] = ['angle_between', 'facing_angle', 'angle_social']
            feats[cam][mouse + mouse2]['social_angle_trig'] = ['sin_angle_between', 'cos_angle_between', 'sin_facing_angle', 'cos_facing_angle', 'sin_angle_social', 'cos_angle_social']
            feats[cam][mouse+mouse2]['relative_size'] = ['area_ellipse_ratio']
            feats[cam][mouse+mouse2]['social_distance'] = ['dist_centroid', 'dist_nose', 'dist_head', 'dist_body',
                                                   'dist_head_body', 'dist_gap', 'dist_scaled', 'overlap_bboxes']

    for cam, parts in zip(cameras.keys(), [cameras[i] for i in cameras.keys()]):
        for mouse in mice:
            feats[cam][mouse]['raw_coordinates'] = [(p + c) for p in parts for c in ['_x', '_y']]
            [feats[cam][mouse]['raw_coordinates'].append(p + c) for p in inferred_parts for c in ['_x', '_y']]
            feats[cam][mouse]['intramouse_distance'] = [('dist_' + p + '_' + q) for p in parts for q in parts if q != p]
        for mouse2 in pairmice:
            feats[cam][mouse+mouse2]['intermouse_distance'] = [('dist_m1' + p + '_m2' + q) for p in parts for q in parts]

    # make sure we have a lambda for each named feature
    for cam in cameras:
        for mouse in feats[cam].keys():
            for grp in feats[cam][mouse].keys():
                feats[cam][mouse][grp] = [f for f in feats[cam][mouse][grp] if f in have_lambdas]

    return feats


def list_features(project):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    feats = generate_valid_feature_list(cfg)
    print('The following feature categories are available:')
    for cam in feats.keys():
        for mouse in feats[cam].keys():
            print("in feats[" + cam + "][" + mouse + "]:")
            for k in feats[cam][mouse].keys():
                print("    '" + k + "'")
            print(' ')
    print('\nFeatures included in each category are:')
    for cam in feats.keys():
        for mouse in feats[cam].keys():
            for feat in feats[cam][mouse].keys():
                print(cam + '|' + mouse + '|' + feat + ':')
                print("   {'" + "', '".join(feats[cam][mouse][feat]) + "'}")
        print(' ')


def flatten_feats(feats, use_grps=[], use_cams=[], use_mice=[]):
    features = []
    for cam in use_cams:
        for mouse in use_mice:
            for feat_class in use_grps:  # feats[cam][mouse].keys():
                features = features + ["_".join((cam, mouse, s)) for s in feats[cam][mouse][feat_class]]

    return features


def center_on_mouse(xa, ya, xb, yb, xa0, ya0, xa00, ya00, boxa, boxb, xlims, ylims):
    # Translate and rotate points so that the neck of mouse A is at the origin, and ears are flat.
    ori_x = xa[3]
    ori_y = ya[3]
    phi = np.arctan2(ya[2] - ya[1], xa[2] - xa[1])
    lam_x = lambda x, y:  (x - ori_x) * np.cos(-phi) + (y - ori_y) * np.sin(-phi)
    lam_y = lambda x, y: -(x - ori_x) * np.sin(-phi) + (y - ori_y) * np.cos(-phi)

    xa_r = lam_x(xa, ya)
    ya_r = lam_y(xa, ya)
    xb_r = lam_x(xb, yb)
    yb_r = lam_y(xb, yb)
    xa0_r = lam_x(xa0, ya0)
    ya0_r = lam_y(xa0, ya0)
    xa00_r = lam_x(xa00, ya00)
    ya00_r = lam_y(xa00, ya00)
    boxa_r = [lam_x(boxa[0], boxa[1]), lam_y(boxa[0], boxa[1]),
              lam_x(boxa[2], boxa[3]), lam_y(boxa[2], boxa[3])]
    boxb_r = [lam_x(boxb[0], boxb[1]), lam_y(boxb[0], boxb[1]),
              lam_x(boxb[2], boxb[3]), lam_y(boxb[2], boxb[3])]
    xlims_r = xlims - ori_x
    ylims_r = ylims - ori_y

    return xa_r, ya_r, xb_r, yb_r, xa0_r, ya0_r, xa00_r, ya00_r, boxa_r, boxb_r, xlims_r, ylims_r

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

def get_mars_keypoints(keypoints, num_mice, partorder):
    xraw = []
    yraw = []
    for m in range(num_mice):
        xraw.append(np.asarray(keypoints[m][0]))
        yraw.append(np.asarray(keypoints[m][1]))
    xm = []
    ym = []
    for m in range(num_mice):
        xm.append(np.array([]))
        ym.append(np.array([]))
        for part in partorder:
            xm[m] = np.append(xm[m], np.mean(xraw[m][part]))
            ym[m] = np.append(ym[m], np.mean(yraw[m][part]))
    return xm, ym


def run_feature_extraction(top_pose_fullpath, opts, progress_bar_sig=[], features=[],
                           front_video_fullpath='', mouse_list=[], center_mouse=False, use_cam='top', max_frames=-1):

    # TODO: this function has a couple optional flags that aren't yet accessible to users:
    # smooth_keypoints - smooth keypoint trajectories before feature extraction (code not actually in place yet)
    # center_mouse - centers the first mouse prior to feature extraction, ie egocentric coordinates

    frames_pose = load_pose(top_pose_fullpath)
    keypoints = [f for f in frames_pose['keypoints']]

    # TODO: add option to smooth keypoint trajectories before feature extraction
    # if smooth_keypoints:
        # keypoints = smooth_keypoint_trajectories(keypoints)

    dscale = opts['pixels_per_cm']
    fps = opts['framerate']
    cfg = opts['classifier_features']['project_config']  # unpack the MARS_developer project config info
    use_grps = features if features else opts['classifier_features']['feat_list'] if 'feat_list' in opts['classifier_features'].keys() else None
    num_frames = len(keypoints)
    if max_frames >= 0:
        num_frames = min(num_frames, max_frames)
    num_mice = len(cfg['animal_names'])*cfg['num_obj']
    if not mouse_list:
        mouse_list = ['m' + str(i) for i in range(num_mice)]

    parts = cfg['keypoints']
    nose       = [parts.index(i) for i in cfg['mars_name_matching']['nose']]
    left_ear   = [parts.index(i) for i in cfg['mars_name_matching']['left_ear']]
    right_ear  = [parts.index(i) for i in cfg['mars_name_matching']['right_ear']]
    neck       = [parts.index(i) for i in cfg['mars_name_matching']['neck']]
    left_side  = [parts.index(i) for i in cfg['mars_name_matching']['left_side']]
    right_side = [parts.index(i) for i in cfg['mars_name_matching']['right_side']]
    tail       = [parts.index(i) for i in cfg['mars_name_matching']['tail']]
    # num_parts = len(parts)
    num_parts = 7  # for now we're just supporting the MARS-style keypoints
    partorder = [nose, right_ear, left_ear, neck, right_side, left_side, tail]

    feats = generate_valid_feature_list(cfg)
    lam = mars_lambdas.generate_lambdas()
    if not use_grps:
        use_grps = []
        for mouse in mouse_list:
            use_grps = use_grps + list(feats[use_cam][mouse].keys())
    else:
        for grp in use_grps:
            if grp not in feats[use_cam][mouse_list[0]].keys():
                raise Exception(grp+' is not a valid feature group name.')
    use_grps.sort()
    features = flatten_feats(feats, use_grps=use_grps, use_cams=[use_cam], use_mice=mouse_list)
    num_features = len(features)
    features_ordered = []

    try:
        bar = progressbar.ProgressBar(widgets=
                                      [progressbar.FormatLabel('Feats frame %(value)d'), '/',
                                       progressbar.FormatLabel('%(max)d  '), progressbar.Percentage(), ' -- ', ' [',
                                       progressbar.Timer(), '] ',
                                       progressbar.Bar(), ' (', progressbar.ETA(), ') '], maxval=num_frames - 1)
        bar.start()

        track = {'features': features,
                 'data': np.zeros((num_mice, num_frames, num_features)),
                 'bbox': np.zeros((num_mice, 4, num_frames)),
                 'keypoints': keypoints,
                 'fps': fps}

        # get some scaling parameters ###################################################
        mouse_length = np.zeros(num_frames)
        allx = []
        ally = []
        for f in range(num_frames):
            keypoints = frames_pose['keypoints'][f]
            xm, ym = get_mars_keypoints(keypoints, num_mice, partorder)

            [allx.append(x) for x in np.ravel(xm)]
            [ally.append(y) for y in np.ravel(ym)]
            mouse_length[f] = np.linalg.norm((xm[0][3] - xm[0][6], ym[0][3] - ym[0][6]))

        # estimate the extent of our arena from tracking data
        allx = np.asarray(allx)/dscale
        ally = np.asarray(ally)/dscale
        xlims_0 = [np.percentile(allx, 1), np.percentile(allx, 99)]
        ylims_0 = [np.percentile(ally, 1), np.percentile(ally, 99)]
        xm0 = [np.array([]) for i in range(num_mice)]
        ym0 = [np.array([]) for i in range(num_mice)]
        xm00 = [np.array([]) for i in range(num_mice)]
        ym00 = [np.array([]) for i in range(num_mice)]

        # extract features ##############################################################
        pr = np.linspace(0, num_frames - 1, num_frames)  # for tracking progress
        for f in range(num_frames):
            bar.update(pr[f])

            if progress_bar_sig:
                if f <= 1:
                    progress_bar_sig.emit(f, num_frames - 1)
                progress_bar_sig.emit(f, 0)

            if f > 1:
                for m in range(num_mice):
                    xm00[m] = xm0[m]
                    ym00[m] = ym0[m]
            if f != 0:
                for m in range(num_mice):
                    xm0[m] = xm[m]
                    ym0[m] = ym[m]

            keypoints = frames_pose['keypoints'][f]
            xm, ym = get_mars_keypoints(keypoints, num_mice, partorder)

            bboxes = []
            for m in range(num_mice):
                bboxes.append(np.asarray(frames_pose['bbox'][f])[0, :])
            if f == 0:
                for m in range(num_mice):
                    xm0[m] = xm[m]
                    ym0[m] = ym[m]
            if f <= 1:
                for m in range(num_mice):
                    xm00[m] = xm0[m]
                    ym00[m] = ym0[m]

            mouse_vals = []
            if num_mice > 1:
                for mouse1 in range(num_mice):
                    for mouse2 in range(num_mice):
                        if mouse2 == mouse1:
                            continue
                        mouse_vals.append(('m'+str(mouse1), 'm'+str(mouse2), xm[mouse1], ym[mouse1], xm[mouse2], ym[mouse2], xm0[mouse1], ym0[mouse1], xm00[mouse1], ym00[mouse1], bboxes[mouse1], bboxes[mouse2]))
            else:
                mouse_vals.append(('m0', '', xm[0], ym[0], xm[0], ym[0], xm0[0], ym0[0], xm00[0], ym00[0], bboxes[0], bboxes[0]))
            for m, (maStr, mbStr, xa, ya, xb, yb, xa0, ya0, xa00, ya00, boxa, boxb) in enumerate(mouse_vals):
                if center_mouse:
                    (xa, ya, xb, yb, xa0, ya0, xa00, ya00, boxa, boxb, xlims, ylims) = \
                        center_on_mouse(xa, ya, xb, yb, xa0, ya0, xa00, ya00, boxa, boxb, xlims_0, ylims_0)
                else:
                    xlims = xlims_0
                    ylims = ylims_0

                # single-mouse features. Lambda returns pixels, convert to cm.
                for feat in lam['xy'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xy'][feat](xa, ya) / dscale

                # single-mouse angle or ratio features. No unit conversion needed.
                for feat in lam['xy_ang'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xy_ang'][feat](xa, ya)

                # ellipse-based features. Lambda returns pixels, convert to cm.
                ell = fit_ellipse(xa, ya)
                for feat in lam['ell'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['ell'][feat](ell) / dscale

                # ellipse-based angle or ratio features. No unit conversion needed.
                for feat in lam['ell_ang'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['ell_ang'][feat](ell)

                # ellipse-based area features. Lambda returns pixels^2, convert to cm^2.
                for feat in lam['ell_area'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['ell_area'][feat](ell) / (dscale ** 2)

                # velocity features. Lambda returns pix/frame, convert to cm/second.
                for feat in lam['dt'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['dt'][feat](xa, ya, xa0, ya0) * fps / dscale

                # acceleration features. Lambda returns pix/frame^2, convert to cm/second^2.
                for feat in lam['d2t'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = \
                            lam['d2t'][feat](xa, ya, xa0, ya0, xa00, ya00) * fps * fps / dscale

                if num_mice > 1:
                    # two-mouse features. Lambda returns pixels, convert to cm.
                    for feat in lam['xyxy'].keys():
                        featname = "_".join((use_cam, mouse_list[m], feat))
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = lam['xyxy'][feat](xa, ya, xb, yb) / dscale

                    # two-mouse angle or ratio features. No unit conversion needed.
                    for feat in lam['xyxy_ang'].keys():
                        featname = "_".join((use_cam, mouse_list[m], feat))
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = lam['xyxy_ang'][feat](xa, ya, xb, yb)

                    # two-mouse velocity features. Lambda returns pix/frame, convert to cm/second.
                    for feat in lam['2mdt'].keys():
                        featname = "_".join((use_cam, mouse_list[m], feat))
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = \
                                lam['2mdt'][feat](xa, ya, xa0, ya0, xb, yb) * fps / dscale

                # Bounding box features. No unit conversion needed so far.
                for feat in lam['bb'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['bb'][feat](boxa, boxb)

                # environment-based features. Lambda returns pixels, convert to cm.
                for feat in lam['xybd'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xybd'][feat](xa, ya, xlims, ylims) / dscale

        # TODO: we could apply smoothing here if we wanted.
        # track['features'] = features_ordered
        track['data_smooth'] = track['data']
        del track['data']

        bar.finish()
        return track

    except Exception as e:
        import linecache
        print("Error when extracting features:")
        exc_type, exc_obj, tb = sys.exc_info()
        filename = tb.tb_frame.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, tb.tb_lineno, tb.tb_frame.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, tb.tb_lineno, line.strip(), exc_obj))
        print(e)
        return []


def compute_windows_features(features, view, featToKeep, windows=[3, 11, 21], num_mice=2):
    feats_name = np.array(features['features'])
    features = features['data_smooth']

    # cleanup and normalization of pixel-derived features
    features = clean_data(features)
    features = normalize_pixel_data(features, view)
    features = clean_data(features)

    # concatenate features from each mouse if necessary
    if num_mice == 1:
        features = features[0, :, :]
    else:
        keepList = [range(np.shape(features)[2])]
        keepList.extend(featToKeep*(num_mice-1))
        features = np.concatenate([features[i, :, ind].transpose() for i, ind in zip(range(num_mice), keepList)], axis=1)
        feats_name = np.concatenate([[str(i) + '_' + f for f in feats_name(inds,)]
                                     for i, inds in zip(range(num_mice, keepList))]).tolist()

    data_win = compute_JAABA_feats(features, windows)
    feats_wnd_names = []
    fn = ['min', 'max', 'mean', 'std']
    for f in feats_name:
        for w in windows:
            for x in fn:
                feats_wnd_names.append('_'.join([f, str(w), x]))
    features_wnd = {'data': data_win,
                    'features': feats_wnd_names}

    return features_wnd


def extract_features_wrapper(opts, video_fullpath, progress_bar_sig='', output_suffix='', front_video_fullpath=''):

    doOverwrite = opts['doOverwrite']
    max_frames = opts['max_frames']
    video_path = os.path.dirname(video_fullpath)
    video_name = os.path.basename(video_fullpath)
    output_folder = mof.get_mouse_output_dir(dir_output_should_be_in=video_path, video_name=video_name, output_suffix=output_suffix)

    feature_view = 'top' if opts['hasTopCamera'] and opts['hasFrontCamera'] else 'top' if opts['hasTopCamera'] else 'front' if opts['hasFrontCamera'] else ''

    pose_basename = mof.get_pose_no_ext(video_fullpath=video_fullpath, output_folder=output_folder, view='top', output_suffix=output_suffix)
    feat_basename_dict = mof.get_feat_no_ext(opts, video_fullpath=video_fullpath, view=feature_view, output_folder=output_folder, output_suffix=output_suffix)
    clf_models = mof.get_classifier_list(opts['classifier_model'])
    top_pose_fullpath = pose_basename + '.json'
    try:
        if not os.path.exists(top_pose_fullpath):
            raise ValueError("No pose has been extracted for this video!")

        feature_types_extracted = []
        feat_from_all_behaviors = {'features': [], 'data_smooth': False, 'bbox': False, 'keypoints': False, 'fps': []}
        featFlag = False

        for behavior in feat_basename_dict.keys():
            feat_basename = feat_basename_dict[behavior]['path']
            feature_type = feat_basename_dict[behavior]['feature_type']
            use_grps = feat_basename_dict[behavior]['feature_groups']
            cfg = feat_basename_dict[behavior]['clf_config']

            if feature_type == 'custom':
                num_mice = len(cfg['animal_names']) * cfg['num_obj']
                mouse_list = ['m' + str(i) for i in range(num_mice)]
                all_feats = generate_valid_feature_list(cfg)
                feature_names = flatten_feats(all_feats, use_grps=use_grps, use_cams=[feature_view], use_mice=mouse_list)

                if os.path.exists(feat_basename + '.npz'):  # we may have features in the right format already, but we have to make sure they contain everything we want for this behavior
                    if not feature_types_extracted and not doOverwrite:  # we haven't extracted features this run, but we have some in a file from earlier, and we're not overwriting it.
                        feat_from_all_behaviors = np.load(feat_basename + '.npz')
                        existing_features = feat_from_all_behaviors['features']
                        featFlag = True
                        feature_types_extracted = existing_features.tolist()
                    # check the feature types we've extracted so far:
                    if all(f in feature_types_extracted for f in feature_names):
                        if not doOverwrite:
                            continue
                features_to_add = [f for f in feature_names if f not in feature_types_extracted]
                grps_to_add = list(set([g for m in mouse_list for g in list(all_feats[feature_view][m].keys()) for f in features_to_add if f.replace(feature_view+'_', '').replace(m+'_', '') in all_feats[feature_view][m][g]]))
                feature_types_extracted += features_to_add
                feature_types_extracted = list(set(feature_types_extracted))
                if not grps_to_add:
                    continue

            if (not os.path.exists(feat_basename + '.npz')) | doOverwrite:
                t = time.time()

                model_name = mof.get_most_recent(opts['classifier_model'], clf_models, behavior)
                clf = joblib.load(os.path.join(opts['classifier_model'], model_name))
                opts['classifier_features'] = clf['params']  # pass along the settings for this classifier

                if feature_type == 'custom':
                    cfg = clf['params']['project_config']  # unpack the MARS_developer parent project config
                    num_mice = len(cfg['animal_names']) * cfg['num_obj']
                    feat = run_feature_extraction(top_pose_fullpath=top_pose_fullpath,
                                                  opts=opts,
                                                  progress_bar_sig=progress_bar_sig,
                                                  max_frames=max_frames,
                                                  features=grps_to_add)
                    feat['features'] = feat_from_all_behaviors['features'] + feat['features']  # 'features' field reflects features in order added
                    feat['data_smooth'] = np.concatenate((feat_from_all_behaviors['data_smooth'], feat['data_smooth']), axis=2) if featFlag else feat['data_smooth']
                    feat_from_all_behaviors = copy.deepcopy(feat)
                    featFlag = True

                elif feature_type == 'raw_pcf':
                    num_mice = 2
                    feat = mlf.classic_extract_features_top_pcf(top_video_fullpath=video_fullpath,
                                                            front_video_fullpath=front_video_fullpath,
                                                            top_pose_fullpath=top_pose_fullpath,
                                                            progress_bar_sig=progress_bar_sig,
                                                            max_frames=max_frames)

                elif feature_type == 'raw':
                    num_mice = 2
                    feat = mlf.classic_extract_features_top(top_video_fullpath=video_fullpath,
                                                        top_pose_fullpath=top_pose_fullpath,
                                                        progress_bar_sig=progress_bar_sig,
                                                        max_frames=max_frames)
                else:
                    raise ValueError("feature type " + feature_type + "not recognized")

                if type(feat) is not dict:
                    raise ValueError('Feature extraction failed for behavior ' + behavior + ', feature type ' + feature_type)
                else:
                    np.savez(feat_basename, **feat)
                    sp.savemat(feat_basename + '.mat',  feat)

                    # do windowing, if we're using old-style MARS features (in newer version we wait til classification time to window)
                    n_feat = feat['data_smooth'].shape[2]

                    if feature_type == 'custom':
                        featToKeep = tuple(flatten([range(n_feat)]))
                        view = 'custom'
                    elif feature_type == 'raw_pcf':
                        featToKeep = tuple(flatten([range(39), range(50, 66), 67, 69, 70, 71, range(121, n_feat)]))
                        view = feature_view + '_pcf'
                        windows=[int(np.ceil(w * opts['framerate']) * 2 + 1) for w in [0.033333, 0.16667, 0.33333]]
                        feat_wnd = compute_windows_features(feat, view, featToKeep, windows=windows, num_mice=num_mice)
                        np.savez(feat_basename + "_wnd", **feat_wnd)
                        sp.savemat(feat_basename + '_wnd.mat', feat_wnd)
                    elif feature_type == 'raw':
                        featToKeep = tuple(flatten([range(39), range(42, 58), 59, 61, 62, 63, range(113, n_feat)]))
                        view = feature_view
                        windows = [int(np.ceil(w * opts['framerate']) * 2 + 1) for w in [0.033333, 0.16667, 0.33333]]
                        feat_wnd = compute_windows_features(feat, view, featToKeep, windows=windows, num_mice=num_mice)
                        np.savez(feat_basename + "_wnd", **feat_wnd)
                        sp.savemat(feat_basename + '_wnd.mat', feat_wnd)
                    elif feature_type != 'custom':
                        raise ValueError("feature type " + feature_type + "not recognized")

            else:
                print('2 - Features top already extracted')

        dt = (time.time() - t) / 60.
        print('[DONE] feature extraction in %5.2f mins' % (dt))
        return
    except Exception as e:
        import linecache
        print("Error when extracting features (extract_top_features_wrapper):")
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
        print(e)
        return []
