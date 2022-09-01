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
            feats[cam][mouse]['joint_angle'] = ['angle_head_body_l', 'angle_head_body_r']
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
                        if m == 0 and f == 0:
                            features_ordered.append(featname)

                # single-mouse angle or ratio features. No unit conversion needed.
                for feat in lam['xy_ang'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xy_ang'][feat](xa, ya)
                        if m == 0 and f == 0:
                            features_ordered.append(featname)

                # ellipse-based features. Lambda returns pixels, convert to cm.
                ell = fit_ellipse(xa, ya)
                for feat in lam['ell'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['ell'][feat](ell) / dscale
                        if m == 0 and f == 0:
                            features_ordered.append(featname)

                # ellipse-based angle or ratio features. No unit conversion needed.
                for feat in lam['ell_ang'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['ell_ang'][feat](ell)
                        if m == 0 and f == 0:
                            features_ordered.append(featname)

                # ellipse-based area features. Lambda returns pixels^2, convert to cm^2.
                for feat in lam['ell_area'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['ell_area'][feat](ell) / (dscale ** 2)
                        if m == 0 and f == 0:
                            features_ordered.append(featname)

                # velocity features. Lambda returns pix/frame, convert to cm/second.
                for feat in lam['dt'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['dt'][feat](xa, ya, xa0, ya0) * fps / dscale
                        if m == 0 and f == 0:
                            features_ordered.append(featname)

                # acceleration features. Lambda returns pix/frame^2, convert to cm/second^2.
                for feat in lam['d2t'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = \
                            lam['d2t'][feat](xa, ya, xa0, ya0, xa00, ya00) * fps * fps / dscale
                        if m == 0 and f == 0:
                            features_ordered.append(featname)

                if num_mice > 1:
                    # two-mouse features. Lambda returns pixels, convert to cm.
                    for feat in lam['xyxy'].keys():
                        featname = "_".join((use_cam, mouse_list[m], feat))
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = lam['xyxy'][feat](xa, ya, xb, yb) / dscale
                            if m == 0 and f == 0:
                                features_ordered.append(featname)

                    # two-mouse angle or ratio features. No unit conversion needed.
                    for feat in lam['xyxy_ang'].keys():
                        featname = "_".join((use_cam, mouse_list[m], feat))
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = lam['xyxy_ang'][feat](xa, ya, xb, yb)
                            if m == 0 and f == 0:
                                features_ordered.append(featname)

                    # two-mouse velocity features. Lambda returns pix/frame, convert to cm/second.
                    for feat in lam['2mdt'].keys():
                        featname = "_".join((use_cam, mouse_list[m], feat))
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = \
                                lam['2mdt'][feat](xa, ya, xa0, ya0, xb, yb) * fps / dscale
                            if m == 0 and f == 0:
                                features_ordered.append(featname)

                # Bounding box features. No unit conversion needed so far.
                for feat in lam['bb'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['bb'][feat](boxa, boxb)
                        if m == 0 and f == 0:
                            features_ordered.append(featname)

                # environment-based features. Lambda returns pixels, convert to cm.
                for feat in lam['xybd'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xybd'][feat](xa, ya, xlims, ylims) / dscale
                        if m == 0 and f == 0:
                            features_ordered.append(featname)

        # TODO: we could apply smoothing here if we wanted.
        track['features'] = features_ordered
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


def classic_extract_features_top(top_video_fullpath,top_pose_fullpath, progress_bar_sig='', max_frames=-1):
    video_name = os.path.basename(top_video_fullpath)
    frames_pose = load_pose(top_pose_fullpath)
    num_points = len(frames_pose['scores'][0][0])
    # align frames
    ext = top_video_fullpath[-3:]

    reader = vidReader(top_video_fullpath)
    num_frames = reader.NUM_FRAMES
    im_h = reader.IM_H
    im_w = reader.IM_W
    fps = reader.fps
    if max_frames >= 0:
        num_frames = min(num_frames, max_frames)

    pix_per_cm = 37.79527606671214
    eps = np.spacing(1)
    parts = ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base']

    features = [
        'nose_x', 'nose_y',
        'right_ear_x', 'right_ear_y',
        'left_ear_x', 'left_ear_y',
        'neck_x', 'neck_y',
        'right_side_x', 'right_side_y',
        'left_side_x', 'left_side_y',
        'tail_base_x', 'tail_base_y',

        'centroid_x', 'centroid_y',
        'centroid_head_x', 'centroid_head_y',
        'centroid_hips_x', 'centroid_hips_y',
        'centroid_body_x', 'centroid_body_y',
        'phi',
        'ori_head',
        'ori_body',
        'angle_head_body_l',
        'angle_head_body_r',
        'major_axis_len',
        'minor_axis_len',
        'axis_ratio',
        'area_ellipse',
        'dist_edge_x',
        'dist_edge_y',
        'dist_edge',

        'speed',
        'speed_centroid',
        'acceleration',
        'acceleration_centroid',
        'speed_fwd',

        # pixel based features
        'resh_twd_itrhb',
        'pixel_change_ubbox_mice',
        'pixel_change',        # patches
        'nose_pc', 'right_ear_pc', 'left_ear_pc', 'neck_pc', 'right_side_pc', 'left_side_pc', 'tail_base_pc',

        'rel_angle_social',  # Angle of heading relative to other animal (PNAS Feat. 12, Eyrun's Feat. 11) 12? not used
        # 'rel_angle',  # direction difference btw mice (Eyrun's Feat. 11) not used
        'rel_dist_gap',  # distance as defined in PNAS (Feat. 13)
        'rel_dist_scaled',  # distance in units of semiaxis (Feat. 14)
        'rel_dist_centroid',
        'rel_dist_nose',
        'rel_dist_head',
        'rel_dist_body',
        'rel_dist_head_body',
        'rel_dist_centroid_change',  # not used
        'overlap_bboxes',
        'area_ellipse_ratio',
        'angle_between',
        'facing_angle',
        'radial_vel',
        'tangential_vel',

        # inter distances
        'dist_m1nose_m2nose',
        'dist_m1nose_m2right_ear',
        'dist_m1nose_m2left_ear',
        'dist_m1nose_m2neck',
        'dist_m1nose_m2right_side',
        'dist_m1nose_m2left_side',
        'dist_m1nose_m2tail_base',
        'dist_m1right_ear_m2nose',
        'dist_m1right_ear_m2right_ear',
        'dist_m1right_ear_m2left_ear',
        'dist_m1right_ear_m2neck',
        'dist_m1right_ear_m2right_side',
        'dist_m1right_ear_m2left_side',
        'dist_m1right_ear_m2tail_base',
        'dist_m1left_ear_m2nose',
        'dist_m1left_ear_m2right_ear',
        'dist_m1left_ear_m2left_ear',
        'dist_m1left_ear_m2neck',
        'dist_m1left_ear_m2right_side',
        'dist_m1left_ear_m2left_side',
        'dist_m1left_ear_m2tail_base',
        'dist_m1neck_m2nose',
        'dist_m1neck_m2right_ear',
        'dist_m1neck_m2left_ear',
        'dist_m1neck_m2neck',
        'dist_m1neck_m2right_side',
        'dist_m1neck_m2left_side',
        'dist_m1neck_m2tail_base',
        'dist_m1right_side_m2nose',
        'dist_m1right_side_m2right_ear',
        'dist_m1right_side_m2left_ear',
        'dist_m1right_side_m2neck',
        'dist_m1right_side_m2right_side',
        'dist_m1right_side_m2left_side',
        'dist_m1right_side_m2tail_base',
        'dist_m1left_side_m2nose',
        'dist_m1left_side_m2right_ear',
        'dist_m1left_side_m2left_ear',
        'dist_m1left_side_m2neck',
        'dist_m1left_side_m2right_side',
        'dist_m1left_side_m2left_side',
        'dist_m1left_side_m2tail_base',
        'dist_m1tail_base_m2nose',
        'dist_m1tail_base_m2right_ear',
        'dist_m1tail_base_m2left_ear',
        'dist_m1tail_base_m2neck',
        'dist_m1tail_base_m2right_side',
        'dist_m1tail_base_m2left_side',
        'dist_m1tail_base_m2tail_base',

        # intra distance
        'dist_nose_right_ear',
        'dist_nose_left_ear',
        'dist_nose_neck',
        'dist_nose_right_side',
        'dist_nose_left_side',
        'dist_nose_tail_base',
        'dist_right_ear_left_ear',
        'dist_right_ear_neck',
        'dist_right_ear_right_side',
        'dist_right_ear_left_side',
        'dist_right_ear_tail_base',
        'dist_left_ear_neck',
        'dist_left_ear_right_side',
        'dist_left_ear_left_side',
        'dist_left_ear_tail_base',
        'dist_neck_right_side',
        'dist_neck_left_side',
        'dist_neck_tail_base',
        'dist_right_side_left_side',
        'dist_right_side_tail_base',
        'dist_left_side_tail_base',

        # velocity centoird w 2 5 10
        'speed_centroid_w2',
        'speed_centroid_w5',
        'speed_centroid_w10',
        # velocity allpoints w2 w5 w10
        'speed_nose_w2',
        'speed_nose_w5',
        'speed_nose_w10',
        'speed_right_ear_w2',
        'speed_right_ear_w5',
        'speed_right_ear_w10',
        'speed_left_ear_w2',
        'speed_left_ear_w5',
        'speed_left_ear_w10',
        'speed_neck_w2',
        'speed_neck_w5',
        'speed_neck_w10',
        'speed_right_side_w2',
        'speed_right_side_w5',
        'speed_right_side_w10',
        'speed_left_side_w2',
        'speed_left_side_w5',
        'speed_left_side_w10',
        'speed_tail_base_w2',
        'speed_tail_base_w5',
        'speed_tail_base_w10'
    ]

    num_features = len(features)
    keypoints = [f for f in frames_pose['keypoints']]

    smooth_kernel = np.array([1, 2, 1]) / 4.

    try:
        bar = progressbar.ProgressBar(widgets=
                                      [progressbar.FormatLabel('Feats frame %(value)d'), '/', progressbar.FormatLabel('%(max)d  '), progressbar.Percentage(), ' -- ', ' [', progressbar.Timer(), '] ',
                                       progressbar.Bar(), ' (', progressbar.ETA(), ') '], maxval=num_frames-1)
        bar.start()
        track = {
            'vid_name': video_name,
            'features': features,
            'data': np.zeros((2, num_frames, num_features)),
            'data_smooth': np.zeros((2, num_frames, num_features)),
            'bbox': np.zeros((2, 4, num_frames)),
            'keypoints': keypoints,
            'fps': fps
        }

        allx, ally = [], []
        ############################################### collecting data from bbox and keypoints
        for f in range(num_frames):
            track['bbox'][:, :, f] = np.asarray(frames_pose['bbox'][f])
            # track['bbox_front'][:,:,f] = np.asarray(frames_pose_front['bbox'][f])
            keypoints = frames_pose['keypoints'][f]
            if 'conf_nose' in features:
                track['data'][0][f][14:21] = frames_pose['scores'][f][0]
                track['data'][1][f][14:21] = frames_pose['scores'][f][1]
            # mouse 1 data
            xm1 = np.asarray(keypoints[0][0])
            ym1 = np.asarray(keypoints[0][1])
            m1k = np.asarray(keypoints[0]).T
            track['data'][0, f, :14] = np.asarray(keypoints[0]).T.flatten()

            # mouse 2 data
            xm2 = np.asarray(keypoints[1][0])
            ym2 = np.asarray(keypoints[1][1])
            m2k = np.asarray(keypoints[1]).T
            track['data'][1, f, :14] = np.asarray(keypoints[1]).T.flatten()

            allx.append(xm1)
            allx.append(xm2)
            ally.append(ym1)
            ally.append(ym2)

            #################################################################  position features

            # fit ellipse
            # ell = {'cx': cx, 'cy': cy, 'ra': a, 'rb': b, 'phi': phi,
            #        'xs': xs, 'ys': ys, 'ori_vec_v': ori_vec_v, 'ori_vec_h': ori_vec_h}
            ell = fit_ellipse(xm1, ym1)
            cx1 = ell['cx']
            cy1 = ell['cy']
            ra1 = ell['ra']
            rb1 = ell['rb']
            alpha1 = ell['phi']
            ind = features.index('centroid_x')
            track['data'][0, f, ind] = cx1  # 'centroid_x'
            ind = features.index('centroid_y')
            track['data'][0, f, ind] = cy1  # 'centroid_y'
            ind = features.index('phi')
            track['data'][0, f, ind] = alpha1  # 'phi'
            ind = features.index('major_axis_len')
            track['data'][0, f, ind] = ra1 if ra1 > 0. else eps  # 'major_axis_len'
            ind = features.index('minor_axis_len')
            track['data'][0, f, ind] = rb1 if rb1 > 0. else eps  # 'minor_axis_len'
            ind = features.index('axis_ratio')
            track['data'][0, f, ind] = ra1 / rb1 if rb1 > 0. else eps  # 'axis_ratio'
            ind = features.index('area_ellipse')
            track['data'][0, f, ind] = mh.pi * ra1 * rb1 if ra1 * rb1 > 0. else eps  # 'area_ellipse'

            ell = fit_ellipse(xm2, ym2)
            cx2 = ell['cx']
            cy2 = ell['cy']
            ra2 = ell['ra']
            rb2 = ell['rb']
            alpha2 = ell['phi']
            ind = features.index('centroid_x')
            track['data'][1, f, ind] = cx2  # 'centroid_x'
            ind = features.index('centroid_y')
            track['data'][1, f, ind] = cy2  # 'centroid_y'
            ind = features.index('phi')
            track['data'][1, f, ind] = alpha2  # 'phi'
            ind = features.index('major_axis_len')
            track['data'][1, f, ind] = ra2 if ra2 > 0. else eps  # 'major_axis_len'
            ind = features.index('minor_axis_len')
            track['data'][1, f, ind] = rb2 if rb2 > 0. else eps  # 'minor_axis_len'
            ind = features.index('axis_ratio')
            track['data'][1, f, ind] = ra2 / rb2 if rb2 > 0. else eps  # 'axis_ratio'
            ind = features.index('area_ellipse')
            track['data'][1, f, ind] = mh.pi * ra2 * rb2 if ra2 * rb2 > 0. else eps  # 'area_ellipse'

            # overlap bbox
            ind = features.index('overlap_bboxes')
            track['data'][0, f, ind] = bb_intersection_over_union(track['bbox'][0, :, f], track['bbox'][1, :, f], im_w, im_h)
            track['data'][1, f, ind] = bb_intersection_over_union(track['bbox'][1, :, f], track['bbox'][0, :, f], im_w, im_h)

            # compute ori head (neck to nose)
            # orim1_head  =
            ind = features.index('ori_head')
            track['data'][0, f, ind] = get_angle(xm1[3], ym1[3], xm1[0], ym1[0])
            # orim2_head  =
            track['data'][1, f, ind] = get_angle(xm2[3], ym2[3], xm2[0], ym2[0])

            # compute ori body (tail to neck)
            # orim1_body  =
            ind = features.index('ori_body')
            track['data'][0, f, ind] = get_angle(xm1[6], ym1[6], xm1[3], ym1[3])
            # orim2_body  =
            track['data'][1, f, ind] = get_angle(xm2[6], ym2[6], xm2[3], ym2[3])

            # compute angle betwwen head and body
            ind = features.index('angle_head_body_l')
            track['data'][0, f, ind] = interior_angle([xm1[2], ym1[2]], [xm1[3], ym1[3]], [xm1[5], ym1[5]])
            track['data'][1, f, ind] = interior_angle([xm2[2], ym2[2]], [xm2[3], ym2[3]], [xm2[5], ym2[5]])
            ind = features.index('angle_head_body_r')
            track['data'][0, f, ind] = interior_angle([xm1[1], ym1[1]], [xm1[3], ym1[3]], [xm1[4], ym1[4]])
            track['data'][1, f, ind] = interior_angle([xm2[1], ym2[1]], [xm2[3], ym2[3]], [xm2[4], ym2[4]])
            # track['data'][0, f, ind] = get_angle(np.mean(xm1[4:7]), np.mean(ym1[4:7]), np.mean(xm1[:3]), np.mean(ym1[:3]))
            # track['data'][1, f, ind] = get_angle(np.mean(xm2[4:7]), np.mean(ym2[4:7]), np.mean(xm2[:3]), np.mean(ym2[:3]))

            # centroid head
            ind = features.index('centroid_head_x')
            track['data'][0, f, ind] = np.mean(xm1[:3])
            track['data'][1, f, ind] = np.mean(xm2[:3])
            ind = features.index('centroid_head_y')
            track['data'][0, f, ind] = np.mean(ym1[:3])
            track['data'][1, f, ind] = np.mean(ym2[:3])

            # centroid hips
            ind = features.index('centroid_hips_x')
            track['data'][0, f, ind] = np.mean(xm1[4:7])
            track['data'][1, f, ind] = np.mean(xm2[4:7])
            ind = features.index('centroid_hips_y')
            track['data'][0, f, ind] = np.mean(ym1[4:7])
            track['data'][1, f, ind] = np.mean(ym2[4:7])

            # centroid body
            ind = features.index('centroid_body_x')
            track['data'][0, f, ind] = np.mean(xm1[3:7])
            track['data'][1, f, ind] = np.mean(xm2[3:7])
            ind = features.index('centroid_body_y')
            track['data'][0, f, ind] = np.mean(ym1[3:7])
            track['data'][1, f, ind] = np.mean(ym2[3:7])

        for f in range(num_features):
            if f == features.index('phi') or f == features.index('ori_head') or f == features.index('ori_body') \
                    or f == features.index('angle_head_body_l') or f == features.index('angle_head_body_r'):
                # track['data_smooth'][:, :, f] = np.mod(track['data_smooth'][:, :, f] + 2*np.pi,2*np.pi)
                track['data_smooth'][:, :, f] = track['data'][:, :, f]
            else:
                track['data_smooth'][0, :, f] = track['data'][0, :, f]
                track['data_smooth'][0, 1:-1, f] = sig.convolve(track['data_smooth'][0, :, f], smooth_kernel, 'valid')
                track['data_smooth'][1, :, f] = track['data'][1, :, f]
                track['data_smooth'][1, 1:-1, f] = sig.convolve(track['data_smooth'][1, :, f], smooth_kernel, 'valid')
        ############################################### features

        # velocity
        # Metric of speed that accounts for both the head movement and the body movement
        # Also this isn't the speed calculated in the PNAS paper (that's speed_fwd)
        if 'speed' in features:
            ind = features.index('speed')
            chx1 = track['data'][0, :, features.index('centroid_head_x')]
            chy1 = track['data'][0, :, features.index('centroid_head_y')]
            chix1 = track['data'][0, :, features.index('centroid_hips_x')]
            chiy1 = track['data'][0, :, features.index('centroid_hips_y')]
            chx2 = track['data'][1, :, features.index('centroid_head_x')]
            chy2 = track['data'][1, :, features.index('centroid_head_y')]
            chix2 = track['data'][1, :, features.index('centroid_hips_x')]
            chiy2 = track['data'][1, :, features.index('centroid_hips_y')]
            x_diff0 = np.diff(np.vstack((chx1, chix1)))
            y_diff0 = np.diff(np.vstack((chy1, chiy1)))
            x_diff1 = np.diff(np.vstack((chx2, chix2)))
            y_diff1 = np.diff(np.vstack((chy2, chiy2)))
            vel1 = np.linalg.norm([x_diff0, y_diff0], axis=(0, 1))
            vel2 = np.linalg.norm([x_diff1, y_diff1], axis=(0, 1))
            track['data'][0, :, ind] = np.hstack((vel1[0], vel1)) * fps / pix_per_cm
            track['data'][1, :, ind] = np.hstack((vel2[0], vel2)) * fps / pix_per_cm

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            chix1[1:-1] = sig.convolve(chix1, smooth_kernel, 'valid')
            chiy1[1:-1] = sig.convolve(chiy1, smooth_kernel, 'valid')
            chix2[1:-1] = sig.convolve(chix2, smooth_kernel, 'valid')
            chiy2[1:-1] = sig.convolve(chiy2, smooth_kernel, 'valid')
            x_diff0 = np.diff(np.vstack((chx1, chix1)))
            y_diff0 = np.diff(np.vstack((chy1, chiy1)))
            x_diff1 = np.diff(np.vstack((chx2, chix2)))
            y_diff1 = np.diff(np.vstack((chy2, chiy2)))
            vel1 = np.linalg.norm([x_diff0, y_diff0], axis=(0, 1))
            vel2 = np.linalg.norm([x_diff1, y_diff1], axis=(0, 1))
            track['data_smooth'][0, :, ind] = np.hstack((vel1[0], vel1)) * fps / pix_per_cm
            track['data_smooth'][1, :, ind] = np.hstack((vel2[0], vel2)) * fps / pix_per_cm

        # forward speed
        if 'speed_fwd' in features:
            ind = features.index('speed_fwd')
            cent_x_ind = features.index('centroid_x')
            cent_y_ind = features.index('centroid_y')
            dir_mot = np.zeros((2, num_frames))
            cx0 = track['data'][0, :, cent_x_ind]
            cy0 = track['data'][0, :, cent_y_ind]
            cx1 = track['data'][1, :, cent_x_ind]
            cy1 = track['data'][1, :, cent_y_ind]
            nskip = 1
            for f in range(num_frames - nskip):
                dir_mot[0, f] = get_angle(cx0[f], cy0[f], cx0[f + nskip], cy0[f + nskip])
                dir_mot[1, f] = get_angle(cx1[f], cy1[f], cx1[f + nskip], cy1[f + nskip])
            for f in range(nskip):
                dir_mot[0, num_frames - nskip + f] = 0
                dir_mot[1, num_frames - nskip + f] = 0
            t_step = 4
            filt = np.ones(t_step)
            x_diff0 = sig.correlate(np.concatenate((np.zeros(1), np.diff(track['data'][0, :, cent_x_ind]))), filt, mode='same')
            y_diff0 = sig.correlate(np.concatenate((np.zeros(1), np.diff(track['data'][0, :, cent_y_ind]))), filt, mode='same')
            x_diff1 = sig.correlate(np.concatenate((np.zeros(1), np.diff(track['data'][1, :, cent_x_ind]))), filt, mode='same')
            y_diff1 = sig.correlate(np.concatenate((np.zeros(1), np.diff(track['data'][1, :, cent_y_ind]))), filt, mode='same')
            ori_ind = features.index('ori_body')
            cos_fac_0 = np.cos(track['data'][0, :, ori_ind] -dir_mot[0, :])
            cos_fac_1 = np.cos(track['data'][1, :, ori_ind] -dir_mot[1, :])
            fs1 = np.multiply(np.linalg.norm([x_diff0, y_diff0], axis=0), cos_fac_0)
            fs2 = np.multiply(np.linalg.norm([x_diff1, y_diff1], axis=0), cos_fac_1)
            track['data'][0, :, ind] = fs1 * fps / pix_per_cm
            track['data'][1, :, ind] = fs2 * fps / pix_per_cm
            fs1[1:-1] = sig.convolve(fs1, smooth_kernel, 'valid')
            fs2[1:-1] = sig.convolve(fs2, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = fs1 * fps / pix_per_cm
            track['data_smooth'][1, :, ind] = fs2 * fps / pix_per_cm

        # acceleration
        if 'acceleration' in features:
            ind = features.index('acceleration')
            chx1 = track['data'][0, :, features.index('centroid_head_x')]
            chy1 = track['data'][0, :, features.index('centroid_head_y')]
            chix1 = track['data'][0, :, features.index('centroid_hips_x')]
            chiy1 = track['data'][0, :, features.index('centroid_hips_y')]
            chx2 = track['data'][1, :, features.index('centroid_head_x')]
            chy2 = track['data'][1, :, features.index('centroid_head_y')]
            chix2 = track['data'][1, :, features.index('centroid_hips_x')]
            chiy2 = track['data'][1, :, features.index('centroid_hips_y')]
            x_acc = np.diff(np.vstack((chx1, chix1)), n=2)
            y_acc = np.diff(np.vstack((chy1, chiy1)), n=2)
            acc1 = np.linalg.norm([x_acc, y_acc], axis=(0, 1))
            track['data'][0, :, ind] = np.hstack((np.hstack((acc1[0], acc1)), acc1[-1])) * fps / (pix_per_cm ** 2)
            x_acc = np.diff(np.vstack((chx2, chix2)), n=2)
            y_acc = np.diff(np.vstack((chy2, chiy2)), n=2)
            acc2 = np.linalg.norm([x_acc, y_acc], axis=(0, 1))
            track['data'][1, :, ind] = np.hstack((np.hstack((acc2[0], acc2)), acc2[-1])) * fps / (pix_per_cm ** 2)

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            chix1[1:-1] = sig.convolve(chix1, smooth_kernel, 'valid')
            chiy1[1:-1] = sig.convolve(chiy1, smooth_kernel, 'valid')
            chix2[1:-1] = sig.convolve(chix2, smooth_kernel, 'valid')
            chiy2[1:-1] = sig.convolve(chiy2, smooth_kernel, 'valid')
            x_acc = np.diff(np.vstack((chx1, chix1)), n=2)
            y_acc = np.diff(np.vstack((chy1, chiy1)), n=2)
            acc1 = np.linalg.norm([x_acc, y_acc], axis=(0, 1))
            track['data_smooth'][0, :, ind] = np.hstack((np.hstack((acc1[0], acc1)), acc1[-1])) * fps / (pix_per_cm ** 2)
            x_acc = np.diff(np.vstack((chx2, chix2)), n=2)
            y_acc = np.diff(np.vstack((chy2, chiy2)), n=2)
            acc2 = np.linalg.norm([x_acc, y_acc], axis=(0, 1))
            track['data_smooth'][1, :, ind] = np.hstack((np.hstack((acc2[0], acc2)), acc2[-1])) * fps / (pix_per_cm ** 2)

        # distance between mice's centroids
        if 'rel_dist_centroid' in features:
            ind = features.index('rel_dist_centroid')
            cent_x_ind = features.index('centroid_x')
            cent_y_ind = features.index('centroid_y')
            x_dif = track['data'][0, :, cent_x_ind] - track['data'][1, :, cent_x_ind]
            y_dif = track['data'][0, :, cent_y_ind] - track['data'][1, :, cent_y_ind]
            cent_dif = np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)
            track['data'][0, :, ind] = cent_dif / pix_per_cm
            track['data'][1, :, ind] = cent_dif / pix_per_cm
            cent_dif[1:-1] = sig.convolve(cent_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = cent_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = cent_dif / pix_per_cm

        # distance change mice centroids
        if 'rel_dist_centroid_change' in features:
            ind = features.index('rel_dist_centroid_change')
            dist_bet = np.diff(track['data'][0, :, features.index('rel_dist_centroid')] * pix_per_cm)
            rdcc = np.hstack((dist_bet[0], dist_bet))
            track['data'][0, :, ind] = rdcc / pix_per_cm
            track['data'][1, :, ind] = rdcc / pix_per_cm
            rdcc[1:-1] = sig.convolve(rdcc, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = rdcc / pix_per_cm
            track['data_smooth'][1, :, ind] = rdcc / pix_per_cm

        # distance between mice's nose
        if 'rel_dist_nose' in features:
            ind = features.index('rel_dist_nose')
            nose_x_ind = features.index('nose_x')
            nose_y_ind = features.index('nose_y')
            x_dif = track['data'][1, :, nose_x_ind] - track['data'][0, :, nose_x_ind]
            y_dif = track['data'][1, :, nose_y_ind] - track['data'][0, :, nose_y_ind]
            nose_dif = np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)
            track['data'][0, :, ind] = nose_dif / pix_per_cm
            track['data'][1, :, ind] = nose_dif / pix_per_cm
            nose_dif[1:-1] = sig.convolve(nose_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = nose_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = nose_dif / pix_per_cm

        # rel dist mice's head
        if 'rel_dist_head' in features:
            ind = features.index('rel_dist_head')  # centroid_body_x
            h_x_ind = features.index('centroid_head_x')
            h_y_ind = features.index('centroid_head_y')
            x_dif = track['data'][1, :, h_x_ind] - track['data'][0, :, h_x_ind]
            y_dif = track['data'][1, :, h_y_ind] - track['data'][0, :, h_y_ind]
            bod_dif = np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)
            track['data'][0, :, ind] = bod_dif / pix_per_cm
            track['data'][1, :, ind] = bod_dif / pix_per_cm
            bod_dif[1:-1] = sig.convolve(bod_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = bod_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = bod_dif / pix_per_cm

        # distance between mice's body centroids
        if 'rel_dist_body' in features:
            ind = features.index('rel_dist_body')  # centroid_body_x
            bod_x_ind = features.index('centroid_body_x')
            bod_y_ind = features.index('centroid_body_y')
            x_dif = track['data'][1, :, bod_x_ind] - track['data'][0, :, bod_x_ind]
            y_dif = track['data'][1, :, bod_y_ind] - track['data'][0, :, bod_y_ind]
            bod_dif = np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)
            track['data'][0, :, ind] = bod_dif / pix_per_cm
            track['data'][1, :, ind] = bod_dif / pix_per_cm
            bod_dif[1:-1] = sig.convolve(bod_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = bod_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = bod_dif / pix_per_cm

        # distance between mice's head body
        if 'rel_dist_head_body' in features:
            ind = features.index('rel_dist_head_body')  # centroid_body_x
            bod_x_ind = features.index('centroid_body_x')
            bod_y_ind = features.index('centroid_body_y')
            h_x_ind = features.index('centroid_head_x')
            h_y_ind = features.index('centroid_head_y')
            x1_dif = track['data'][0, :, h_x_ind] - track['data'][1, :, bod_x_ind]
            y1_dif = track['data'][0, :, h_y_ind] - track['data'][1, :, bod_y_ind]
            x2_dif = track['data'][1, :, h_x_ind] - track['data'][0, :, bod_x_ind]
            y2_dif = track['data'][1, :, h_y_ind] - track['data'][0, :, bod_y_ind]
            hb1_dif = np.linalg.norm(np.vstack((x1_dif, y1_dif)), axis=0)
            hb2_dif = np.linalg.norm(np.vstack((x2_dif, y2_dif)), axis=0)
            track['data'][0, :, ind] = hb1_dif / pix_per_cm
            track['data'][1, :, ind] = hb2_dif / pix_per_cm
            hb1_dif[1:-1] = sig.convolve(hb1_dif, smooth_kernel, 'valid')
            hb2_dif[1:-1] = sig.convolve(hb2_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = hb1_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = hb2_dif / pix_per_cm

        # Angle of heading relative to other animal (PNAS Feat. 12, Eyrun's Feat. 11)
        if 'rel_angle_social' in features:
            ind = features.index('rel_angle_social')
            bod_x_ind = features.index('centroid_x')
            bod_y_ind = features.index('centroid_y')
            ang_ind = features.index('ori_body')
            x_dif = track['data'][1, :, bod_x_ind] - track['data'][0, :, bod_x_ind]  # r_x
            y_dif = track['data'][1, :, bod_y_ind] - track['data'][0, :, bod_y_ind]  # r_y
            theta0 = (np.arctan2(y_dif, x_dif) + 2 * np.pi) % 2 * np.pi
            theta1 = (np.arctan2(-y_dif, -x_dif) + 2 * np.pi) % 2 * np.pi
            ang0 = np.mod(theta0 - track['data'][0, :, ang_ind], 2 * np.pi)
            ang1 = np.mod(theta1 - track['data'][1, :, ang_ind], 2 * np.pi)
            track['data'][0, :, ind] = np.minimum(ang0, 2 * np.pi - ang0)
            track['data'][1, :, ind] = np.minimum(ang1, 2 * np.pi - ang1)
            track['data_smooth'][0, :, ind] = track['data'][0, :, ind]
            track['data_smooth'][0, 1:-1, ind] = np.mod(sig.convolve(track['data_smooth'][0, :, ind], smooth_kernel, 'valid') + 2 * np.pi, 2 * np.pi)
            track['data_smooth'][1, :, ind] = track['data'][1, :, ind]
            track['data_smooth'][1, 1:-1, ind] = np.mod(sig.convolve(track['data_smooth'][1, :, ind], smooth_kernel, 'valid') + 2 * np.pi, 2 * np.pi)

        # semiaxis length
        M_ind = features.index('major_axis_len')
        m_ind = features.index('minor_axis_len')
        phi_ind = features.index('phi')
        c_M_0 = np.multiply(track['data'][0, :, M_ind] * pix_per_cm, np.sin(track['data'][0, :, phi_ind]))
        c_m_0 = np.multiply(track['data'][0, :, m_ind] * pix_per_cm, np.cos(track['data'][0, :, phi_ind]))
        c_M_1 = np.multiply(track['data'][1, :, M_ind] * pix_per_cm, np.sin(track['data'][1, :, phi_ind]))
        c_m_1 = np.multiply(track['data'][1, :, m_ind] * pix_per_cm, np.cos(track['data'][1, :, phi_ind]))

        # distance as defined in PNAS (Feat. 13)
        if 'rel_dist_gap' in features:
            ind = features.index('rel_dist_gap')
            rel_dist_ind = features.index('rel_dist_body')
            comb_norm = np.linalg.norm(np.vstack((c_M_0, c_m_0)), axis=0) + np.linalg.norm(np.vstack((c_M_1, c_m_1)), axis=0)
            d1 = (track['data'][0, :, rel_dist_ind] * pix_per_cm - comb_norm)
            d2 = (track['data'][1, :, rel_dist_ind] * pix_per_cm - comb_norm)
            track['data'][0, :, ind] = d1 / pix_per_cm
            track['data'][1, :, ind] = d2 / pix_per_cm
            d1[1:-1] = sig.convolve(d1, smooth_kernel, 'valid')
            d2[1:-1] = sig.convolve(d2, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = d1 / pix_per_cm
            track['data_smooth'][1, :, ind] = d2 / pix_per_cm

        # distance in units of semiaxis (Feat. 14)
        if 'rel_dist_scaled' in features:
            ind = features.index('rel_dist_scaled')
            rel_dist_gap_ind = features.index('rel_dist_gap')
            d1 = np.divide(track['data'][0, :, rel_dist_gap_ind] * pix_per_cm, np.linalg.norm(np.vstack((c_M_0, c_m_0)), axis=0))
            d2 = np.divide(track['data'][1, :, rel_dist_gap_ind] * pix_per_cm, np.linalg.norm(np.vstack((c_M_1, c_m_1)), axis=0))
            track['data'][0, :, ind] = d1 / pix_per_cm
            track['data'][1, :, ind] = d2 / pix_per_cm
            d1[1:-1] = sig.convolve(d1, smooth_kernel, 'valid')
            d2[1:-1] = sig.convolve(d2, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = d1 / pix_per_cm
            track['data_smooth'][1, :, ind] = d2 / pix_per_cm

        # Ratio of mouse 1's ellipse area to mouse 0's ellipse area (Feat. 15)
        if 'area_ellipse_ratio' in features:
            ind = features.index('area_ellipse_ratio')
            area_ind = features.index('area_ellipse')
            track['data'][0, :, ind] = np.divide(track['data'][0, :, area_ind], track['data'][1, :, area_ind], dtype=float)
            track['data'][1, :, ind] = np.divide(track['data'][1, :, area_ind], track['data'][0, :, area_ind], dtype=float)
            track['data_smooth'][0, :, ind] = track['data'][0, :, ind]
            track['data_smooth'][0, 1:-1, ind] = sig.convolve(track['data_smooth'][0, :, ind], smooth_kernel, 'valid')
            track['data_smooth'][1, :, ind] = track['data'][1, :, ind]
            track['data_smooth'][1, 1:-1, ind] = sig.convolve(track['data_smooth'][1, :, ind], smooth_kernel, 'valid')

        # angle between & facing angle
        if 'angle_between' in features:
            ind = features.index('angle_between')
            ori_ind = features.index(('phi'))
            vec_rot = np.vstack((np.cos(track['data'][0, :, ori_ind]), -np.sin(track['data'][0, :, ori_ind])))
            vec_rot2 = np.vstack((np.cos(track['data'][1, :, ori_ind]), -np.sin(track['data'][1, :, ori_ind])))
            angle_btw = np.arccos((vec_rot * vec_rot2).sum(axis=0))
            track['data'][0, :, ind] = angle_btw
            track['data'][1, :, ind] = angle_btw
            angle_btw[1:-1] = sig.convolve(angle_btw, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = angle_btw
            track['data_smooth'][1, :, ind] = angle_btw

        if 'facing_angle' in features:
            ori_ind = features.index(('phi'))
            vec_rot = np.vstack((np.cos(track['data'][0, :, ori_ind]), -np.sin(track['data'][0, :, ori_ind])))
            vec_rot2 = np.vstack((np.cos(track['data'][1, :, ori_ind]), -np.sin(track['data'][1, :, ori_ind])))
            c1 = np.vstack((track['data'][0, :, features.index('centroid_x')], track['data'][0, :, features.index('centroid_y')]))
            c2 = np.vstack((track['data'][1, :, features.index('centroid_x')], track['data'][1, :, features.index('centroid_y')]))
            vec_btw = c2 - c1
            norm_btw = np.linalg.norm(np.vstack((vec_btw[0, :], vec_btw[1, :])), axis=0)
            vec_btw = vec_btw / np.repeat([norm_btw], 2, axis=0)
            fa1 = np.arccos((vec_rot * vec_btw).sum(axis=0))
            fa2 = np.arccos((vec_rot2 * -vec_btw).sum(axis=0))
            ind = features.index('facing_angle')
            track['data'][0, :, ind] = fa1
            track['data'][1, :, ind] = fa2
            fa1[1:-1] = sig.convolve(fa1, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = fa1
            fa2[1:-1] = sig.convolve(fa2, smooth_kernel, 'valid')
            track['data_smooth'][1, :, ind] = fa2

        # distance points mouse1 to mouse 2, m1 and m1 , m2 and m2
        if 'dist_m1nose_m2nose' in features or 'dist_nose_right_ear' in features:
            if 'dist_m1nose_m2nose' in features:
                m1m2d = np.zeros((num_frames, num_points ** 2))
                idx_triu = np.triu_indices(num_points, 1)
                n_idx = len(idx_triu[0])
            if 'dist_nose_right_ear' in features:
                m1m1d = np.zeros((num_frames, n_idx))
                m2m2d = np.zeros((num_frames, n_idx))

            for f in range(num_frames):
                m1 = track['data'][0, f, :(num_points * 2)]
                m2 = track['data'][1, f, :(num_points * 2)]
                m1_x = m1[::2]
                m1_y = m1[1::2]
                m2_x = m2[::2]
                m2_y = m2[1::2]
                m1_p = np.vstack((m1_x, m1_y)).T
                m2_p = np.vstack((m2_x, m2_y)).T

                if 'dist_m1nose_m2nose' in features:
                    m1m2d[f, :] = dist.cdist(m1_p, m2_p).flatten()
                if 'dist_nose_right_ear' in features:
                    m1m1d[f, :] = dist.cdist(m1_p, m1_p)[idx_triu]
                    m2m2d[f, :] = dist.cdist(m2_p, m2_p)[idx_triu]

            if 'dist_m1nose_m2nose' in features:
                ind_m1m2d = features.index('dist_m1nose_m2nose')
                track['data'][0, :, ind_m1m2d:ind_m1m2d + num_points ** 2] = m1m2d / pix_per_cm
                track['data'][1, :, ind_m1m2d:ind_m1m2d + num_points ** 2] = m1m2d / pix_per_cm
                for d in range(m1m2d.shape[1]): m1m2d[1:-1, d] = sig.convolve(m1m2d[:, d], smooth_kernel, 'valid')
                track['data_smooth'][0, :, ind_m1m2d:ind_m1m2d + num_points ** 2] = m1m2d / pix_per_cm
                track['data_smooth'][1, :, ind_m1m2d:ind_m1m2d + num_points ** 2] = m1m2d / pix_per_cm
            if 'dist_nose_right_ear' in features:
                ind_md = features.index('dist_nose_right_ear')
                track['data'][0, :, ind_md:ind_md + n_idx] = m1m1d / pix_per_cm
                track['data'][1, :, ind_md:ind_md + n_idx] = m2m2d / pix_per_cm
                for d in range(m1m1d.shape[1]): m1m1d[1:-1, d] = sig.convolve(m1m1d[:, d], smooth_kernel, 'valid')
                for d in range(m2m2d.shape[1]): m2m2d[1:-1, d] = sig.convolve(m2m2d[:, d], smooth_kernel, 'valid')
                track['data_smooth'][0, :, ind_md:ind_md + n_idx] = m1m1d / pix_per_cm
                track['data_smooth'][1, :, ind_md:ind_md + n_idx] = m2m2d / pix_per_cm

        # velocity centroid windows -2 -5 -10
        if 'speed_centroid' in features or 'speed_centroid_x' in features or 'speed_centroid_y' in features:
            chx1 = track['data'][0, :, features.index('centroid_x')]
            chy1 = track['data'][0, :, features.index('centroid_y')]
            chx2 = track['data'][1, :, features.index('centroid_x')]
            chy2 = track['data'][1, :, features.index('centroid_y')]
            x_diff0 = np.diff(chx1)
            y_diff0 = np.diff(chy1)
            x_diff1 = np.diff(chx2)
            y_diff1 = np.diff(chy2)
            if 'speed_centroid' in features:
                ind = features.index('speed_centroid')
                vel0 = np.linalg.norm([x_diff0, y_diff0], axis=0)
                track['data'][0, :, ind] = np.hstack((vel0[0], vel0)) * fps / pix_per_cm
                vel1 = np.linalg.norm([x_diff1, y_diff1], axis=0)
                track['data'][1, :, ind] = np.hstack((vel1[0], vel1)) * fps / pix_per_cm

            if 'speed_centroid_x' in features:
                indx = features.index('speed_centroid_x')
                track['data'][0, :, indx] = np.hstack((x_diff0[0], x_diff0))
                track['data'][1, :, indx] = np.hstack((x_diff1[0], x_diff1))

            if 'speed_centroid_y' in features:
                indy = features.index('speed_centroid_y')
                track['data'][0, :, indy] = np.hstack((y_diff0[0], y_diff0))
                track['data'][1, :, indy] = np.hstack((y_diff1[0], y_diff1))

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            x_diff0 = np.diff(chx1)
            y_diff0 = np.diff(chy1)
            x_diff1 = np.diff(chx2)
            y_diff1 = np.diff(chy2)

            if 'speed_centroid_x' in features:
                track['data_smooth'][0, :, indx] = np.hstack((x_diff0[0], x_diff0))
                track['data_smooth'][1, :, indx] = np.hstack((x_diff1[0], x_diff1))
            if 'speed_centroid_y' in features:
                track['data_smooth'][0, :, indy] = np.hstack((y_diff0[0], y_diff0))
                track['data_smooth'][1, :, indy] = np.hstack((y_diff1[0], y_diff1))
            if 'speed_centroid' in features:
                vel0 = np.linalg.norm([x_diff0, y_diff0], axis=0)
                track['data_smooth'][0, :, ind] = np.hstack((vel0[0], vel0)) * fps / pix_per_cm
                vel1 = np.linalg.norm([x_diff1, y_diff1], axis=0)
                track['data_smooth'][1, :, ind] = np.hstack((vel1[0], vel1)) * fps / pix_per_cm

        # acceleration centroid
        if 'acceleration_centroid' in features or 'acceleration_centroid_x' in features or 'acceleration_centroid_y' in features:
            chx1 = track['data'][0, :, features.index('centroid_x')]
            chy1 = track['data'][0, :, features.index('centroid_y')]
            chx2 = track['data'][1, :, features.index('centroid_x')]
            chy2 = track['data'][1, :, features.index('centroid_y')]
            x_acc0 = np.diff(chx1, n=2)
            y_acc0 = np.diff(chy1, n=2)
            x_acc1 = np.diff(chx2, n=2)
            y_acc1 = np.diff(chy2, n=2)

            if 'acceleration_centroid' in features:
                ind = features.index('acceleration_centroid')
                acc0 = np.linalg.norm([x_acc0, y_acc0], axis=0)
                acc1 = np.linalg.norm([x_acc1, y_acc1], axis=0)
                track['data'][0, :, ind] = np.hstack((acc0[0], acc0[0], acc0)) * fps / (pix_per_cm ** 2)
                track['data'][1, :, ind] = np.hstack((acc1[0], acc1[0], acc1)) * fps / (pix_per_cm ** 2)

            if 'acceleration_centroid_x' in features:
                indx = features.index('acceleration_centroid_x')
                track['data'][0, :, indx] = np.hstack((x_acc0[0], x_acc0[0], x_acc0))
                track['data'][1, :, indx] = np.hstack((x_acc1[0], x_acc1[0], x_acc1))

            if 'acceleration_centroid_y' in features:
                indy = features.index('acceleration_centroid_y')
                track['data'][0, :, indy] = np.hstack((y_acc0[0], y_acc0[0], y_acc0))
                track['data'][1, :, indy] = np.hstack((y_acc1[0], y_acc1[0], y_acc1))

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            x_acc0 = np.diff(chx1, n=2)
            y_acc0 = np.diff(chy1, n=2)
            x_acc1 = np.diff(chx2, n=2)
            y_acc1 = np.diff(chy2, n=2)

            if 'acceleration_centroid' in features:
                acc0 = np.linalg.norm([x_acc0, y_acc0], axis=0)
                acc1 = np.linalg.norm([x_acc1, y_acc1], axis=0)
                track['data_smooth'][0, :, ind] = np.hstack((acc0[0], acc0[0], acc0)) * fps / (pix_per_cm ** 2)
                track['data_smooth'][1, :, ind] = np.hstack((acc1[0], acc1[0], acc1)) * fps / (pix_per_cm ** 2)

            if 'acceleration_centroid_x' in features:
                track['data_smooth'][0, :, indx] = np.hstack((x_acc0[0], x_acc0[0], x_acc0))
                track['data_smooth'][1, :, indx] = np.hstack((x_acc1[0], x_acc1[0], x_acc1))

            if 'acceleration_centroid_y' in features:
                track['data_smooth'][0, :, indy] = np.hstack((y_acc0[0], y_acc0[0], y_acc0))
                track['data_smooth'][1, :, indy] = np.hstack((y_acc1[0], y_acc1[0], y_acc1))

        if 'speed_centroid_w2' in features:
            ind_w2 = features.index('speed_centroid_w2')
            ind_w5 = features.index('speed_centroid_w5')
            ind_w10 = features.index('speed_centroid_w10')
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(2, num_frames - 2):
                x_diff0 = chx1[f] - chx1[f - 2]
                y_diff0 = chy1[f] - chy1[f - 2]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 2]
                y_diff1 = chy2[f] - chy2[f - 2]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data'][0, :, ind_w2] = vel1 * fps / pix_per_cm
            track['data'][1, :, ind_w2] = vel2 * fps / pix_per_cm
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(5, num_frames - 5):
                x_diff0 = chx1[f] - chx1[f - 5]
                y_diff0 = chy1[f] - chy1[f - 5]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 5]
                y_diff1 = chy2[f] - chy2[f - 5]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data'][0, :, ind_w5] = vel1 * fps / pix_per_cm
            track['data'][1, :, ind_w5] = vel2 * fps / pix_per_cm
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(10, num_frames - 10):
                x_diff0 = chx1[f] - chx1[f - 10]
                y_diff0 = chy1[f] - chy1[f - 10]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 10]
                y_diff1 = chy2[f] - chy2[f - 10]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data'][0, :, ind_w10] = vel1 * fps / pix_per_cm
            track['data'][1, :, ind_w10] = vel2 * fps / pix_per_cm

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(2, num_frames - 2):
                x_diff0 = chx1[f] - chx1[f - 2]
                y_diff0 = chy1[f] - chy1[f - 2]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 2]
                y_diff1 = chy2[f] - chy2[f - 2]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data_smooth'][0, :, ind_w2] = vel1 * fps / pix_per_cm
            track['data_smooth'][1, :, ind_w2] = vel2 * fps / pix_per_cm
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(5, num_frames - 5):
                x_diff0 = chx1[f] - chx1[f - 5]
                y_diff0 = chy1[f] - chy1[f - 5]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 5]
                y_diff1 = chy2[f] - chy2[f - 5]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data_smooth'][0, :, ind_w5] = vel1 * fps / pix_per_cm
            track['data_smooth'][1, :, ind_w5] = vel2 * fps / pix_per_cm
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(10, num_frames - 10):
                x_diff0 = chx1[f] - chx1[f - 10]
                y_diff0 = chy1[f] - chy1[f - 10]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 10]
                y_diff1 = chy2[f] - chy2[f - 10]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data_smooth'][0, :, ind_w10] = vel1 * fps / pix_per_cm
            track['data_smooth'][1, :, ind_w10] = vel2 * fps / pix_per_cm

        # velocity points windows -2 -5 -10
        if 'speed_nose_w2' in features:
            for p in range(0, num_points * 2, 2):
                chx1 = track['data'][0, :, p]
                chy1 = track['data'][0, :, p + 1]
                chx2 = track['data'][1, :, p]
                chy2 = track['data'][1, :, p + 1]
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                ind_w2 = features.index('speed_' + parts[int(p / 2)] + '_w2')
                ind_w5 = features.index('speed_' + parts[int(p / 2)] + '_w5')
                ind_w10 = features.index('speed_' + parts[int(p / 2)] + '_w10')
                for f in range(2, num_frames - 2):
                    x_diff0 = chx1[f] - chx1[f - 2]
                    y_diff0 = chy1[f] - chy1[f - 2]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 2]
                    y_diff1 = chy2[f] - chy2[f - 2]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data'][0, :, ind_w2] = vel1 * fps / pix_per_cm
                track['data'][1, :, ind_w2] = vel2 * fps / pix_per_cm
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(5, num_frames - 5):
                    x_diff0 = chx1[f] - chx1[f - 5]
                    y_diff0 = chy1[f] - chy1[f - 5]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 5]
                    y_diff1 = chy2[f] - chy2[f - 5]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data'][0, :, ind_w5] = vel1 * fps / pix_per_cm
                track['data'][1, :, ind_w5] = vel2 * fps / pix_per_cm
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(10, num_frames - 10):
                    x_diff0 = chx1[f] - chx1[f - 10]
                    y_diff0 = chy1[f] - chy1[f - 10]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 10]
                    y_diff1 = chy2[f] - chy2[f - 10]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data'][0, :, ind_w10] = vel1 * fps / pix_per_cm
                track['data'][1, :, ind_w10] = vel2 * fps / pix_per_cm

                chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
                chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
                chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
                chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(2, num_frames - 2):
                    x_diff0 = chx1[f] - chx1[f - 2]
                    y_diff0 = chy1[f] - chy1[f - 2]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 2]
                    y_diff1 = chy2[f] - chy2[f - 2]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data_smooth'][0, :, ind_w2] = vel1 * fps / pix_per_cm
                track['data_smooth'][1, :, ind_w2] = vel2 * fps / pix_per_cm
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(5, num_frames - 5):
                    x_diff0 = chx1[f] - chx1[f - 5]
                    y_diff0 = chy1[f] - chy1[f - 5]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 5]
                    y_diff1 = chy2[f] - chy2[f - 5]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data_smooth'][0, :, ind_w5] = vel1 * fps / pix_per_cm
                track['data_smooth'][1, :, ind_w5] = vel2 * fps / pix_per_cm
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(10, num_frames - 10):
                    x_diff0 = chx1[f] - chx1[f - 10]
                    y_diff0 = chy1[f] - chy1[f - 10]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 10]
                    y_diff1 = chy2[f] - chy2[f - 10]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data_smooth'][0, :, ind_w10] = vel1 * fps / pix_per_cm
                track['data_smooth'][1, :, ind_w10] = vel2 * fps / pix_per_cm

        # distance to edge
        if 'dist_edge_x' in features:
            ind = features.index('dist_edge_x')
            allx = np.concatenate(allx).ravel()
            ally = np.concatenate(ally).ravel()
            allx = allx[allx > 0]
            ally = ally[ally > 0]
            minx = np.percentile(allx, 1)
            maxx = np.percentile(allx, 99)
            miny = np.percentile(ally, 1)
            maxy = np.percentile(ally, 99)

            cx1 = track['data'][0, :, features.index('centroid_x')]
            cx2 = track['data'][1, :, features.index('centroid_x')]
            cy1 = track['data'][0, :, features.index('centroid_y')]
            cy2 = track['data'][1, :, features.index('centroid_y')]
            distsx1 = np.amin(np.stack((np.maximum(0, cx1 - minx), np.maximum(0, maxx - cx1)), axis=-1), axis=1)
            distsx2 = np.amin(np.stack((np.maximum(0, cx2 - minx), np.maximum(0, maxx - cx2)), axis=-1), axis=1)
            distsy1 = np.amin(np.stack((np.maximum(0, cy1 - miny), np.maximum(0, maxy - cy1)), axis=-1), axis=1)
            distsy2 = np.amin(np.stack((np.maximum(0, cy2 - miny), np.maximum(0, maxy - cy2)), axis=-1), axis=1)
            dists1 = np.amin(np.stack((distsx1, distsy1), axis=-1), axis=1)
            dists2 = np.amin(np.stack((distsx2, distsy2), axis=-1), axis=1)
            track['data'][0, :, ind] = distsx1
            track['data'][0, :, ind + 1] = distsy1
            track['data'][0, :, ind + 2] = dists1
            track['data'][1, :, ind] = distsx2
            track['data'][1, :, ind + 1] = distsy2
            track['data'][1, :, ind + 2] = dists2
            # distsx1[1:-1]= sig.convolve(distsx1, smooth_kernel, 'valid')
            # distsx2[1:-1]= sig.convolve(distsx2, smooth_kernel, 'valid')
            # distsy1[1:-1]= sig.convolve(distsy1, smooth_kernel, 'valid')
            # distsy2[1:-1]= sig.convolve(distsy2, smooth_kernel, 'valid')
            # dists1[1:-1]= sig.convolve(dists1, smooth_kernel, 'valid')
            # dists2[1:-1]= sig.convolve(dists2, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = distsx1
            track['data_smooth'][0, :, ind + 1] = distsy1
            track['data_smooth'][0, :, ind + 2] = dists1
            track['data_smooth'][1, :, ind] = distsx2
            track['data_smooth'][1, :, ind + 1] = distsy2
            track['data_smooth'][1, :, ind + 2] = dists2

        # radial vel and tangetial vel
        if 'radial_vel' in features or ' tangential_vel' in features:
            cx1 = track['data'][0, :, features.index('centroid_x')]
            cx2 = track['data'][1, :, features.index('centroid_x')]
            cy1 = track['data'][0, :, features.index('centroid_y')]
            cy2 = track['data'][1, :, features.index('centroid_y')]
            ddx1 = cx2 - cx1
            ddx2 = cx1 - cx2
            ddy1 = cy2 - cy1
            ddy2 = cy1 - cy2
            len1 = np.sqrt(ddx1 ** 2 + ddy1 ** 2)
            len1[len1 == 0] = eps
            len2 = np.sqrt(ddx2 ** 2 + ddy2 ** 2)
            len2[len2 == 0] = eps
            ddx1 = ddx1 / len1
            ddx2 = ddx2 / len2
            ddy1 = ddy1 / len1
            ddy2 = ddy2 / len2
            dx1 = np.diff(cx1)
            dx2 = np.diff(cx2)
            dy1 = np.diff(cy1)
            dy2 = np.diff(cy2)

            if 'radial_vel' in features:
                ind = features.index('radial_vel')
                radial_vel1 = dx1 * ddx1[1:] + dy1 * ddy1[1:]
                radial_vel2 = dx2 * ddx2[1:] + dy2 * ddy2[1:]
                track['data'][0, :, ind] = np.hstack((radial_vel1[0], radial_vel1))
                track['data'][1, :, ind] = np.hstack((radial_vel2[0], radial_vel2))
                radial_vel1[1:-1] = sig.convolve(radial_vel1, smooth_kernel, 'valid')
                radial_vel2[1:-1] = sig.convolve(radial_vel2, smooth_kernel, 'valid')
                track['data_smooth'][0, :, ind] = np.hstack((radial_vel1[0], radial_vel1))
                track['data_smooth'][1, :, ind] = np.hstack((radial_vel2[0], radial_vel2))

            # tangential vel
            if 'tangential_vel' in features:
                ind = features.index('tangential_vel')
                orth_ddx1 = -ddy1
                orth_ddx2 = -ddy2
                orth_ddy1 = -ddx1
                orth_ddy2 = -ddx2
                tang_vel1 = np.abs(dx1 * orth_ddx1[1:] + dy1 * orth_ddy1[1:])
                tang_vel2 = np.abs(dx2 * orth_ddx2[1:] + dy2 * orth_ddy2[1:])
                track['data'][0, :, ind] = np.hstack((tang_vel1[0], tang_vel1))
                track['data'][1, :, ind] = np.hstack((tang_vel2[0], tang_vel2))
                tang_vel1[1:-1] = sig.convolve(tang_vel1, smooth_kernel, 'valid')
                tang_vel2[1:-1] = sig.convolve(tang_vel2, smooth_kernel, 'valid')
                track['data_smooth'][0, :, ind] = np.hstack((tang_vel1[0], tang_vel1))
                track['data_smooth'][1, :, ind] = np.hstack((tang_vel2[0], tang_vel2))


                ################################ pixel based features

        # pixel based features
        if 'pixel_change' in features or 'pixel_change_ubbox_mice' in features or 'pc_nose' in features :

            if 'pixel_change' in features:
                ind = features.index('pixel_change')
            if 'pixel_change_ubbox_mice' in features:
                ind1 = features.index('pixel_change_ubbox_mice')
            if 'nose_pc' in features:
                indn = features.index('nose_pc')

            for f in range(1, num_frames):
                bar.update(f)

                if progress_bar_sig:
                    if f <= 1:
                        progress_bar_sig.emit(f,num_frames-1)
                    progress_bar_sig.emit(f,0)

                if f == 1:
                    frame1 = reader.getFrame(f-1)
                    frame1 = frame1.astype(np.float32)
                else:
                    frame1 = frame2

                frame2 = reader.getFrame(f)
                frame2 = frame2.astype(np.float32)

                if 'pixel_change' in features:
                    track['data'][0, f, ind] = (np.sum((frame2 - frame1) ** 2)) / float((np.sum((frame1) ** 2)))
                    track['data'][1, f, ind] = track['data'][0, f, ind]

                if 'pixel_change_ubbox_mice' in features:
                    ind_bbo = features.index('overlap_bboxes')
                    f1_bb = track['data'][0, f - 1, ind_bbo]
                    f2_bb = track['data'][0, f, ind_bbo]

                    if f1_bb > 0. or f2_bb > 0.:
                        bbox_f1_m1 = track['bbox'][0, :, f - 1]
                        xmin11, xmax11 = bbox_f1_m1[[0, 2]] * im_w
                        ymin11, ymax11 = bbox_f1_m1[[1, 3]] * im_h
                        bbox_f1_m2 = track['bbox'][1, :, f - 1]
                        xmin12, xmax12 = bbox_f1_m2[[0, 2]] * im_w
                        ymin12, ymax12 = bbox_f1_m2[[1, 3]] * im_h

                        bbox_f2_m1 = track['bbox'][0, :, f]
                        xmin21, xmax21 = bbox_f2_m1[[0, 2]] * im_w
                        ymin21, ymax21 = bbox_f2_m1[[1, 3]] * im_h
                        bbox_f2_m2 = track['bbox'][1, :, f]
                        xmin22, xmax22 = bbox_f2_m2[[0, 2]] * im_w
                        ymin22, ymax22 = bbox_f2_m2[[1, 3]] * im_h

                        # xA1 = max(xmin11, xmin12);  yA1 = max(ymin11, ymin12)
                        # xB1 = min(xmax11, xmax12);  yB1 = min(ymax11, ymax12)
                        #
                        # xA2 = max(xmin21, xmin22);  yA2 = max(ymin21, ymin22)
                        # xB2 = min(xmax21, xmax22);  yB2 = min(ymax21, ymax22)

                        # if (xB1 > xA1 and yB1 > yA1) or  (xB2 > xA2 and yB2 > yA2): # there is intersection
                        tmp1 = frame1[int(min(ymin11, ymin12)):int(max(ymax11, ymax12)), int(min(xmin11, xmin12)):int(max(xmax11, xmax12))]
                        tmp2 = frame2[int(min(ymin21, ymin22)):int(max(ymax21, ymax22)), int(min(xmin21, xmin22)):int(max(xmax21, xmax22))]

                        if np.prod(tmp1.shape) > np.prod(tmp2.shape):
                            tmp2 = cv2.resize(tmp2, (tmp1.shape[1], tmp1.shape[0]))
                        else:
                            tmp1 = cv2.resize(tmp1, (tmp2.shape[1], tmp2.shape[0]))

                        track['data'][0, f, ind1] = (np.sum((tmp2 - tmp1) ** 2)) / float((np.sum((tmp1) ** 2)))
                        track['data'][1, f, ind1] = (np.sum((tmp2 - tmp1) ** 2)) / float((np.sum((tmp1) ** 2)))

                if 'nose_pc' in features:
                    for s in range(2):
                        coords = track['data'][s, f - 1, :14]
                        coords2 = track['data'][s, f, :14]

                        xm1 = coords[::2]
                        ym1 = coords[1::2]

                        xm2 = coords2[::2]
                        ym2 = coords2[1::2]
                        ps = 10
                        for p in range(num_points):
                            miny1 = int(ym1[p] - ps)
                            minx1 = int(xm1[p] - ps)
                            maxy1 = int(ym1[p] + ps)
                            maxx1 = int(xm1[p] + ps)
                            while miny1 < 0: miny1 += 1
                            # while maxy1 < 0: maxy1 += 1
                            while minx1 < 0: minx1 += 1
                            # while maxx1 < 0: maxx1 += 1
                            while maxy1 > im_h: maxy1 -= 1
                            # while miny1 > im_h -1 : miny1 -= 1
                            while maxx1 > im_w: maxx1 -= 1
                            # while minx1 > im_w -1: minx1 -= 1

                            miny2 = int(ym2[p] - ps)
                            minx2 = int(xm2[p] - ps)
                            maxy2 = int(ym2[p] + ps)
                            maxx2 = int(xm2[p] + ps)
                            while miny2 < 0: miny2 += 1
                            # while maxy2 < 0: maxy2 += 1
                            while minx2 < 0: minx2 += 1
                            # while maxx2 < 0: maxx2 += 1
                            while maxy2 > im_h: maxy2 -= 1
                            # while miny2 > im_h-1: miny2 -= 1
                            while maxx2 > im_w: maxx2 -= 1
                            # while minx2 > im_w-1: minx2 -= 1

                            f1x = range(minx1, maxx1) if minx1 < maxx1 else range(maxx1, minx1)
                            f1y = range(miny1, maxy1) if miny1 < maxy1 else range(maxy1, miny1)
                            f2x = range(minx2, maxx2) if minx2 < maxx2 else range(maxx2, minx2)
                            f2y = range(miny2, maxy2) if miny2 < maxy2 else range(maxy2, miny2)
                            x1, y1 = np.ix_(f1x, f1y)
                            x2, y2 = np.ix_(f2x, f2y)
                            patch1 = frame1[y1, x1]
                            patch2 = frame2[y2, x2]
                            max_h = max(patch1.shape[0], patch2.shape[0])
                            max_w = max(patch1.shape[1], patch2.shape[1])
                            patch2 = cv2.resize(patch2, (max_w, max_h))
                            patch1 = cv2.resize(patch1, (max_w, max_h))
                            track['data'][s, f, indn + p] = (np.sum((patch2 - patch1) ** 2)) / float((np.sum((patch1) ** 2)))
                            # else:
                            #     track['data'][s, f, ind + p]=0


            if 'pixel_change' in features:
                track['data_smooth'][0, :, ind] = track['data'][0, :, ind]
                track['data_smooth'][0, 1:-1, ind] = sig.convolve(track['data_smooth'][0, :, ind], smooth_kernel, 'valid')
                track['data_smooth'][1, :, ind] = track['data_smooth'][0, :, ind]

            if 'pixel_change_ubbox_mice' in features:
                track['data_smooth'][0, :, ind1] = track['data'][0, :, ind1]
                track['data_smooth'][0, 1:-1, ind1] = sig.convolve(track['data_smooth'][0, :, ind1], smooth_kernel, 'valid')
                track['data_smooth'][1, :, ind1] = track['data'][1, :, ind1]
                track['data_smooth'][1, 1:-1, ind1] = sig.convolve(track['data_smooth'][1, :, ind1], smooth_kernel, 'valid')

            if 'nose_pc' in features:
                for s in range(2):
                    for p in range(num_points):
                        track['data_smooth'][s, :, indn + p] = track['data'][s, :, indn + p]
                        track['data_smooth'][s, 1:-1, indn + p] = sig.convolve(track['data_smooth'][s, :, indn + p], smooth_kernel, 'valid')

        if 'resh_twd_itrhb' in features:
            ind = features.index('resh_twd_itrhb')

            def get_ell_distance(pt, theta):
                # (x,y) coordinate of the nearest point from the intruder's face to an
                # ellipse (intruder's head or body), traveling along a line at angle theta
                # theta = -pi/2 ->  3 o'clock (dead right)
                # theta = 0     -> 12 o'clock (straight ahead)
                # theta = pi/2  ->  9 o'clock (dead left)

                # (xc,yc) -> centroid of ellipse
                # M -> major axis of ellipse (radius)
                # m -> minor axis of ellipse (radius)
                # s -> orientation of ellipse
                # b -> y-intercept of line
                # k -> slope of line
                theta += np.pi / 2.

                b = 0
                k = np.tan(theta)
                xc = np.mean(pt[0, :])
                yc = np.mean(pt[1, :])
                m = distance(pt[:, 1], pt[:, 2]) / 2.
                M = distance(pt[:, 0], pt[:, 3]) / 2.
                s = pt[:, 0] - np.array([xc, yc])
                x, y = intercept_ell(xc, yc, s[1] / s[0], m, M, k, b)

                if (theta > np.pi / 2. and x >= 0) or (theta <= np.pi / 2. and x <= 0):
                    x = np.Inf
                    y = np.Inf

                if not np.isreal(x) or not np.isreal(y):
                    x = np.Inf
                    y = np.Inf

                return np.array([x, y])

            def intercept_ell(xc, yc, s, m, M, k, b):
                A = (k - s)
                B = (b - yc + s * xc)
                C = (1 + s * k)
                D = (s * b - s * yc - xc)
                w1 = m ** 2 * (1 + s ** 2)
                w2 = M ** 2 * (1 + s ** 2)

                aa = (A ** 2 / w1 + C ** 2 / w2)
                bb = (2 * A * B / w1 + 2 * C * D / w2)
                cc = (B ** 2 / w1 + D ** 2 / w2 - 1)

                x1 = (-bb + cmh.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)
                x2 = (-bb - cmh.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)

                y1 = k * x1 + b
                y2 = k * x2 + b

                if (cmh.sqrt(x1 ** 2 + y1 ** 2).real < cmh.sqrt(x2 ** 2 + y2 ** 2).real):
                    x = x1
                    y = y1
                else:
                    x = x2
                    y = y2

                return x, y

            def smooths(vin, wsize, sigma):
                wsize = np.array([1, wsize])
                overshoots = np.floor(wsize[1] / 2).astype(int)
                if overshoots * 2 == wsize[1]:
                    downshoot = overshoots - 1
                    upshoot = overshoots
                else:
                    downshoot = overshoots
                    upshoot = overshoots
                vout = vin
                x = np.arange(-(wsize[1] - 1) / 2, (wsize[1]) / 2)
                smoothfilt = np.exp(-(x * x) / (2 * sigma))
                smoothfilt = smoothfilt / sum(smoothfilt)
                for hdx in range(vin.shape[0]):
                    tvout = np.convolve(vin[hdx, :], smoothfilt, mode='full')
                    vout[hdx, :] = tvout[downshoot:len(tvout) - upshoot]

                return vout

            def distance(p1, p2):
                return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

            ######################
            keypoints = np.array(frames_pose['keypoints'])
            pts = np.zeros((len(keypoints), 2, 15))
            pts[:, :, :7] = keypoints[:, 0, :, :]
            pts[:, :, 7] = (pts[:, :, 1] + pts[:, :, 2]) / 2.
            pts[:, :, 8:] = keypoints[:, 1, :, :]
            pts -= pts[:, :, 7, np.newaxis]
            pts[:, :, 2] = smooths(pts[:, :, 2].transpose(), 125, 25).transpose()

            for fr in range(num_frames):
                # rotate all the points so the resident is facing straight ahead (positive
                # direction on y-axis). I define "straight ahead" as making the line between
                # the ears have zero slope- you could equivalently set it so the line from
                # center-eyes to nose has inf slope (or do 0 slope then rotate an additional 90
                # degrees, to avoid numerical treachery.)
                p = pts[fr]
                th = -mh.atan2(p[1, 2], p[0, 2])
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                p = R.dot(p)

                # compute distance to closest point on intruder mouse's head and body.
                # Works kind of like ray tracing- over a range of theta, ask what the
                # distance from intruder's head to closest point on the resident is, when
                # you travel along a line at orientation theta.
                #
                # theta = -pi/2 ->  3 o'clock (dead right)
                # theta = 0     -> 12 o'clock (straight ahead)
                # theta = pi/2  ->  9 o'clock (dead left)
                # range of angles to check  (can replace this with thR = 0 if you just want to check straight ahead)
                thR = np.arange(-np.pi * 3 / 4, np.pi / 4 + .01, .02)
                head = np.zeros((2, len(thR)))
                body = np.zeros((2, len(thR)))
                for i, t in enumerate(thR):
                    head[:, i] = get_ell_distance(p[:, 8:12], t)
                    body[:, i] = get_ell_distance(p[:, 11:], t)
                # convert coordinates to distances:
                dHead = np.array([distance(np.array([0, 0]), head[:, i]) for i in range(head.shape[1])])
                dBody = np.array([distance(np.array([0, 0]), body[:, i]) for i in range(body.shape[1])])
                dMouse = np.minimum(dHead, dBody)

                # find if there are point of interception
                pp = np.where(dMouse != np.Inf)[0]
                if pp.size: track['data'][:, fr, ind] = 1
                # part = np.zeros(len(dMouse))
                # for i in range(len(dMouse)):    part[i] = 1 if dMouse[i] == dHead[i] else 2
                # int_points = np.zeros((2, len(pp)))
                # for i in range(len(pp)):  int_points[:, i] = head[:, pp[i]] if part[pp[i]] == 1 else body[:, pp[i]]

            track['data_smooth'][:, :, ind] = track['data'][:, :, ind]

        del track['data']
        bar.finish()
        # augment features


        # del track['bbox']
        # del track['bbox_front']
        # del track['keypoints']
        # del track['keypoints_front']
        reader.close()
        return track
    except Exception as e:
        import linecache
        print("Error when extracting features (extract_features_top):")
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
        print(e)
        reader.close()
        return []


def classic_extract_features_top_pcf(top_video_fullpath, front_video_fullpath, top_pose_fullpath, progress_bar_sig='', max_frames=-1):
    video_name = os.path.basename(top_video_fullpath)
    frames_pose = load_pose(top_pose_fullpath)
    num_points = len(frames_pose['scores'][0][0])
    # align frames

    _,ext = os.path.splitext(top_video_fullpath)
    ext=ext[1:]
    reader = vidReader(top_video_fullpath)
    num_frames = reader.NUM_FRAMES
    im_h = reader.IM_H
    im_w = reader.IM_W
    fps = reader.fps
    if max_frames >= 0:
        num_frames = min(num_frames, max_frames)

    _,extf = os.path.splitext(front_video_fullpath)
    extf = extf[1:]
    readerf = vidReader(front_video_fullpath)
    num_framesf = readerf.NUM_FRAMES
    im_wf = readerf.IM_W
    im_hf = readerf.IM_H
    fpsf = readerf.fps

    if max_frames >= 0:
        num_framesf = min(num_framesf, max_frames)

    pix_per_cm = 37.79527606671214
    eps = np.spacing(1)
    parts = ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base']
    flag_align = 1 if abs(num_framesf-num_frames)>3 else 0

    features = [
        'nose_x', 'nose_y',
        'right_ear_x', 'right_ear_y',
        'left_ear_x', 'left_ear_y',
        'neck_x', 'neck_y',
        'right_side_x', 'right_side_y',
        'left_side_x', 'left_side_y',
        'tail_base_x', 'tail_base_y',

        'centroid_x', 'centroid_y',
        'centroid_head_x', 'centroid_head_y',
        'centroid_hips_x', 'centroid_hips_y',
        'centroid_body_x', 'centroid_body_y',
        'phi',
        'ori_head',
        'ori_body',
        'angle_head_body_l',
        'angle_head_body_r',
        'major_axis_len',
        'minor_axis_len',
        'axis_ratio',
        'area_ellipse',
        'dist_edge_x',
        'dist_edge_y',
        'dist_edge',

        'speed',
        'speed_centroid',
        'acceleration',
        'acceleration_centroid',
        'speed_fwd',

        # pixel based features
        'resh_twd_itrhb',
        'pixel_change_ubbox_mice',
        'pixel_change',
        'pixel_change_front',
        'imgDf1', 'imgDf2', 'imgDf3', 'imgDf6', 'imgMean', 'imgStd', 'imgDf1Smooth',
        'nose_pc', 'right_ear_pc', 'left_ear_pc', 'neck_pc', 'right_side_pc', 'left_side_pc', 'tail_base_pc',

        'rel_angle_social',  # Angle of heading relative to other animal (PNAS Feat. 12, Eyrun's Feat. 11) 12? not used
        # 'rel_angle',  # direction difference btw mice (Eyrun's Feat. 11) not used
        'rel_dist_gap',  # distance as defined in PNAS (Feat. 13)
        'rel_dist_scaled',  # distance in units of semiaxis (Feat. 14)
        'rel_dist_centroid',
        'rel_dist_nose',
        'rel_dist_head',
        'rel_dist_body',
        'rel_dist_head_body',
        'rel_dist_centroid_change',  # not used
        'overlap_bboxes',
        'area_ellipse_ratio',
        'angle_between',
        'facing_angle',
        'radial_vel',
        'tangential_vel',

        # inter distances
        'dist_m1nose_m2nose',
        'dist_m1nose_m2right_ear',
        'dist_m1nose_m2left_ear',
        'dist_m1nose_m2neck',
        'dist_m1nose_m2right_side',
        'dist_m1nose_m2left_side',
        'dist_m1nose_m2tail_base',
        'dist_m1right_ear_m2nose',
        'dist_m1right_ear_m2right_ear',
        'dist_m1right_ear_m2left_ear',
        'dist_m1right_ear_m2neck',
        'dist_m1right_ear_m2right_side',
        'dist_m1right_ear_m2left_side',
        'dist_m1right_ear_m2tail_base',
        'dist_m1left_ear_m2nose',
        'dist_m1left_ear_m2right_ear',
        'dist_m1left_ear_m2left_ear',
        'dist_m1left_ear_m2neck',
        'dist_m1left_ear_m2right_side',
        'dist_m1left_ear_m2left_side',
        'dist_m1left_ear_m2tail_base',
        'dist_m1neck_m2nose',
        'dist_m1neck_m2right_ear',
        'dist_m1neck_m2left_ear',
        'dist_m1neck_m2neck',
        'dist_m1neck_m2right_side',
        'dist_m1neck_m2left_side',
        'dist_m1neck_m2tail_base',
        'dist_m1right_side_m2nose',
        'dist_m1right_side_m2right_ear',
        'dist_m1right_side_m2left_ear',
        'dist_m1right_side_m2neck',
        'dist_m1right_side_m2right_side',
        'dist_m1right_side_m2left_side',
        'dist_m1right_side_m2tail_base',
        'dist_m1left_side_m2nose',
        'dist_m1left_side_m2right_ear',
        'dist_m1left_side_m2left_ear',
        'dist_m1left_side_m2neck',
        'dist_m1left_side_m2right_side',
        'dist_m1left_side_m2left_side',
        'dist_m1left_side_m2tail_base',
        'dist_m1tail_base_m2nose',
        'dist_m1tail_base_m2right_ear',
        'dist_m1tail_base_m2left_ear',
        'dist_m1tail_base_m2neck',
        'dist_m1tail_base_m2right_side',
        'dist_m1tail_base_m2left_side',
        'dist_m1tail_base_m2tail_base',

        # intra distance
        'dist_nose_right_ear',
        'dist_nose_left_ear',
        'dist_nose_neck',
        'dist_nose_right_side',
        'dist_nose_left_side',
        'dist_nose_tail_base',
        'dist_right_ear_left_ear',
        'dist_right_ear_neck',
        'dist_right_ear_right_side',
        'dist_right_ear_left_side',
        'dist_right_ear_tail_base',
        'dist_left_ear_neck',
        'dist_left_ear_right_side',
        'dist_left_ear_left_side',
        'dist_left_ear_tail_base',
        'dist_neck_right_side',
        'dist_neck_left_side',
        'dist_neck_tail_base',
        'dist_right_side_left_side',
        'dist_right_side_tail_base',
        'dist_left_side_tail_base',

        # velocity centoird w 2 5 10
        'speed_centroid_w2',
        'speed_centroid_w5',
        'speed_centroid_w10',
        # velocity allpoints w2 w5 w10
        'speed_nose_w2',
        'speed_nose_w5',
        'speed_nose_w10',
        'speed_right_ear_w2',
        'speed_right_ear_w5',
        'speed_right_ear_w10',
        'speed_left_ear_w2',
        'speed_left_ear_w5',
        'speed_left_ear_w10',
        'speed_neck_w2',
        'speed_neck_w5',
        'speed_neck_w10',
        'speed_right_side_w2',
        'speed_right_side_w5',
        'speed_right_side_w10',
        'speed_left_side_w2',
        'speed_left_side_w5',
        'speed_left_side_w10',
        'speed_tail_base_w2',
        'speed_tail_base_w5',
        'speed_tail_base_w10',

    ]

    num_features = len(features)
    keypoints = [f for f in frames_pose['keypoints']]

    num_frames = min(num_frames, num_framesf)
    smooth_kernel = np.array([1, 2, 1]) / 4.

    try:
        bar = progressbar.ProgressBar(widgets=
                                      [progressbar.FormatLabel('Feats frame %(value)d'), '/', progressbar.FormatLabel('%(max)d  '), progressbar.Percentage(), ' -- ', ' [', progressbar.Timer(), '] ',
                                       progressbar.Bar(), ' (', progressbar.ETA(), ') '], maxval=num_frames-1)
        bar.start()
        track = {
            'vid_name': video_name,
            'features': features,
            'data': np.zeros((2, num_frames, num_features)),
            'data_smooth': np.zeros((2, num_frames, num_features)),
            'bbox': np.zeros((2, 4, num_frames)),
            'keypoints': keypoints,
            'fps': fps
        }

        allx, ally = [], []
        ############################################### collecting data from bbox and keypoints
        for f in range(num_frames):
            track['bbox'][:, :, f] = np.asarray(frames_pose['bbox'][f])
            # track['bbox_front'][:,:,f] = np.asarray(frames_pose_front['bbox'][f])
            keypoints = frames_pose['keypoints'][f]
            if 'conf_nose' in features:
                track['data'][0][f][14:21] = frames_pose['scores'][f][0]
                track['data'][1][f][14:21] = frames_pose['scores'][f][1]
            # mouse 1 data
            xm1 = np.asarray(keypoints[0][0])
            ym1 = np.asarray(keypoints[0][1])
            m1k = np.asarray(keypoints[0]).T
            track['data'][0, f, :14] = np.asarray(keypoints[0]).T.flatten()

            # mouse 2 data
            xm2 = np.asarray(keypoints[1][0])
            ym2 = np.asarray(keypoints[1][1])
            m2k = np.asarray(keypoints[1]).T
            track['data'][1, f, :14] = np.asarray(keypoints[1]).T.flatten()

            allx.append(xm1)
            allx.append(xm2)
            ally.append(ym1)
            ally.append(ym2)

            #################################################################  position features

            # fit ellipse
            ell = fit_ellipse(xm1, ym1)
            cx1 = ell['cx']
            cy1 = ell['cy']
            ra1 = ell['ra']
            rb1 = ell['rb']
            alpha1 = ell['phi']
            ind = features.index('centroid_x')
            track['data'][0, f, ind] = cx1  # 'centroid_x'
            ind = features.index('centroid_y')
            track['data'][0, f, ind] = cy1  # 'centroid_y'
            ind = features.index('phi')
            track['data'][0, f, ind] = alpha1  # 'phi'
            ind = features.index('major_axis_len')
            track['data'][0, f, ind] = ra1 if ra1 > 0. else eps  # 'major_axis_len'
            ind = features.index('minor_axis_len')
            track['data'][0, f, ind] = rb1 if rb1 > 0. else eps  # 'minor_axis_len'
            ind = features.index('axis_ratio')
            track['data'][0, f, ind] = ra1 / rb1 if rb1 > 0. else eps  # 'axis_ratio'
            ind = features.index('area_ellipse')
            track['data'][0, f, ind] = mh.pi * ra1 * rb1 if ra1 * rb1 > 0. else eps  # 'area_ellipse'

            ell = fit_ellipse(xm2, ym2)
            cx2 = ell['cx']
            cy2 = ell['cy']
            ra2 = ell['ra']
            rb2 = ell['rb']
            alpha2 = ell['phi']
            ind = features.index('centroid_x')
            track['data'][1, f, ind] = cx2  # 'centroid_x'
            ind = features.index('centroid_y')
            track['data'][1, f, ind] = cy2  # 'centroid_y'
            ind = features.index('phi')
            track['data'][1, f, ind] = alpha2  # 'phi'
            ind = features.index('major_axis_len')
            track['data'][1, f, ind] = ra2 if ra2 > 0. else eps  # 'major_axis_len'
            ind = features.index('minor_axis_len')
            track['data'][1, f, ind] = rb2 if rb2 > 0. else eps  # 'minor_axis_len'
            ind = features.index('axis_ratio')
            track['data'][1, f, ind] = ra2 / rb2 if rb2 > 0. else eps  # 'axis_ratio'
            ind = features.index('area_ellipse')
            track['data'][1, f, ind] = mh.pi * ra2 * rb2 if ra2 * rb2 > 0. else eps  # 'area_ellipse'

            # overlap bbox
            ind = features.index('overlap_bboxes')
            track['data'][0, f, ind] = bb_intersection_over_union(track['bbox'][0, :, f], track['bbox'][1, :, f], im_w, im_h)
            track['data'][1, f, ind] = bb_intersection_over_union(track['bbox'][1, :, f], track['bbox'][0, :, f], im_w, im_h)

            # compute ori head (neck to nose)
            # orim1_head  =
            ind = features.index('ori_head')
            track['data'][0, f, ind] = get_angle(xm1[3], ym1[3], xm1[0], ym1[0])
            # orim2_head  =
            track['data'][1, f, ind] = get_angle(xm2[3], ym2[3], xm2[0], ym2[0])

            # compute ori body (tail to neck)
            # orim1_body  =
            ind = features.index('ori_body')
            track['data'][0, f, ind] = get_angle(xm1[6], ym1[6], xm1[3], ym1[3])
            # orim2_body  =
            track['data'][1, f, ind] = get_angle(xm2[6], ym2[6], xm2[3], ym2[3])

            # compute angle betwwen head and body
            ind = features.index('angle_head_body_l')
            track['data'][0, f, ind] = interior_angle([xm1[2], ym1[2]], [xm1[3], ym1[3]], [xm1[5], ym1[5]])
            track['data'][1, f, ind] = interior_angle([xm2[2], ym2[2]], [xm2[3], ym2[3]], [xm2[5], ym2[5]])
            ind = features.index('angle_head_body_r')
            track['data'][0, f, ind] = interior_angle([xm1[1], ym1[1]], [xm1[3], ym1[3]], [xm1[4], ym1[4]])
            track['data'][1, f, ind] = interior_angle([xm2[1], ym2[1]], [xm2[3], ym2[3]], [xm2[4], ym2[4]])
            # track['data'][0, f, ind] = get_angle(np.mean(xm1[4:7]), np.mean(ym1[4:7]), np.mean(xm1[:3]), np.mean(ym1[:3]))
            # track['data'][1, f, ind] = get_angle(np.mean(xm2[4:7]), np.mean(ym2[4:7]), np.mean(xm2[:3]), np.mean(ym2[:3]))

            # centroid head
            ind = features.index('centroid_head_x')
            track['data'][0, f, ind] = np.mean(xm1[:3])
            track['data'][1, f, ind] = np.mean(xm2[:3])
            ind = features.index('centroid_head_y')
            track['data'][0, f, ind] = np.mean(ym1[:3])
            track['data'][1, f, ind] = np.mean(ym2[:3])

            # centroid hips
            ind = features.index('centroid_hips_x')
            track['data'][0, f, ind] = np.mean(xm1[4:7])
            track['data'][1, f, ind] = np.mean(xm2[4:7])
            ind = features.index('centroid_hips_y')
            track['data'][0, f, ind] = np.mean(ym1[4:7])
            track['data'][1, f, ind] = np.mean(ym2[4:7])

            # centroid body
            ind = features.index('centroid_body_x')
            track['data'][0, f, ind] = np.mean(xm1[3:7])
            track['data'][1, f, ind] = np.mean(xm2[3:7])
            ind = features.index('centroid_body_y')
            track['data'][0, f, ind] = np.mean(ym1[3:7])
            track['data'][1, f, ind] = np.mean(ym2[3:7])

        for f in range(num_features):
            if f == features.index('phi') or f == features.index('ori_head') or f == features.index('ori_body') \
                    or f == features.index('angle_head_body_l') or f == features.index('angle_head_body_r'):
                # track['data_smooth'][:, :, f] = np.mod(track['data_smooth'][:, :, f] + 2*np.pi,2*np.pi)
                track['data_smooth'][:, :, f] = track['data'][:, :, f]
            else:
                track['data_smooth'][0, :, f] = track['data'][0, :, f]
                track['data_smooth'][0, 1:-1, f] = sig.convolve(track['data_smooth'][0, :, f], smooth_kernel, 'valid')
                track['data_smooth'][1, :, f] = track['data'][1, :, f]
                track['data_smooth'][1, 1:-1, f] = sig.convolve(track['data_smooth'][1, :, f], smooth_kernel, 'valid')

        ############################################### features

        # velocity
        # Metric of speed that accounts for both the head movement and the body movement
        # Also this isn't the speed calculated in the PNAS paper (that's speed_fwd)
        if 'speed' in features:
            ind = features.index('speed')
            chx1 = track['data'][0, :, features.index('centroid_head_x')]
            chy1 = track['data'][0, :, features.index('centroid_head_y')]
            chix1 = track['data'][0, :, features.index('centroid_hips_x')]
            chiy1 = track['data'][0, :, features.index('centroid_hips_y')]
            chx2 = track['data'][1, :, features.index('centroid_head_x')]
            chy2 = track['data'][1, :, features.index('centroid_head_y')]
            chix2 = track['data'][1, :, features.index('centroid_hips_x')]
            chiy2 = track['data'][1, :, features.index('centroid_hips_y')]
            x_diff0 = np.diff(np.vstack((chx1, chix1)))
            y_diff0 = np.diff(np.vstack((chy1, chiy1)))
            x_diff1 = np.diff(np.vstack((chx2, chix2)))
            y_diff1 = np.diff(np.vstack((chy2, chiy2)))
            vel1 = np.linalg.norm([x_diff0, y_diff0], axis=(0, 1))
            vel2 = np.linalg.norm([x_diff1, y_diff1], axis=(0, 1))
            track['data'][0, :, ind] = np.hstack((vel1[0], vel1)) * fps / pix_per_cm
            track['data'][1, :, ind] = np.hstack((vel2[0], vel2)) * fps / pix_per_cm

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            chix1[1:-1] = sig.convolve(chix1, smooth_kernel, 'valid')
            chiy1[1:-1] = sig.convolve(chiy1, smooth_kernel, 'valid')
            chix2[1:-1] = sig.convolve(chix2, smooth_kernel, 'valid')
            chiy2[1:-1] = sig.convolve(chiy2, smooth_kernel, 'valid')
            x_diff0 = np.diff(np.vstack((chx1, chix1)))
            y_diff0 = np.diff(np.vstack((chy1, chiy1)))
            x_diff1 = np.diff(np.vstack((chx2, chix2)))
            y_diff1 = np.diff(np.vstack((chy2, chiy2)))
            vel1 = np.linalg.norm([x_diff0, y_diff0], axis=(0, 1))
            vel2 = np.linalg.norm([x_diff1, y_diff1], axis=(0, 1))
            track['data_smooth'][0, :, ind] = np.hstack((vel1[0], vel1)) * fps / pix_per_cm
            track['data_smooth'][1, :, ind] = np.hstack((vel2[0], vel2)) * fps / pix_per_cm

        # forward speed
        if 'speed_fwd' in features:
            ind = features.index('speed_fwd')
            cent_x_ind = features.index('centroid_x')
            cent_y_ind = features.index('centroid_y')
            dir_mot = np.zeros((2, num_frames))
            cx0 = track['data'][0, :, cent_x_ind]
            cy0 = track['data'][0, :, cent_y_ind]
            cx1 = track['data'][1, :, cent_x_ind]
            cy1 = track['data'][1, :, cent_y_ind]
            nskip = 1
            for f in range(num_frames - nskip):
                dir_mot[0, f] = get_angle(cx0[f], cy0[f], cx0[f + nskip], cy0[f + nskip])
                dir_mot[1, f] = get_angle(cx1[f], cy1[f], cx1[f + nskip], cy1[f + nskip])
            for f in range(nskip):
                dir_mot[0, num_frames - nskip + f] = 0
                dir_mot[1, num_frames - nskip + f] = 0
            t_step = 4
            filt = np.ones(t_step)
            x_diff0 = sig.correlate(np.concatenate((np.zeros(1), np.diff(track['data'][0, :, cent_x_ind]))), filt, mode='same')
            y_diff0 = sig.correlate(np.concatenate((np.zeros(1), np.diff(track['data'][0, :, cent_y_ind]))), filt, mode='same')
            x_diff1 = sig.correlate(np.concatenate((np.zeros(1), np.diff(track['data'][1, :, cent_x_ind]))), filt, mode='same')
            y_diff1 = sig.correlate(np.concatenate((np.zeros(1), np.diff(track['data'][1, :, cent_y_ind]))), filt, mode='same')
            ori_ind = features.index('ori_body')
            cos_fac_0 = np.cos(track['data'][0, :, ori_ind] -dir_mot[0, :])
            cos_fac_1 = np.cos(track['data'][1, :, ori_ind] -dir_mot[1, :])
            fs1 = np.multiply(np.linalg.norm([x_diff0, y_diff0], axis=0), cos_fac_0)
            fs2 = np.multiply(np.linalg.norm([x_diff1, y_diff1], axis=0), cos_fac_1)
            track['data'][0, :, ind] = fs1 * fps / pix_per_cm
            track['data'][1, :, ind] = fs2 * fps / pix_per_cm
            fs1[1:-1] = sig.convolve(fs1, smooth_kernel, 'valid')
            fs2[1:-1] = sig.convolve(fs2, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = fs1 * fps / pix_per_cm
            track['data_smooth'][1, :, ind] = fs2 * fps / pix_per_cm

        # acceleration
        if 'acceleration' in features:
            ind = features.index('acceleration')
            chx1 = track['data'][0, :, features.index('centroid_head_x')]
            chy1 = track['data'][0, :, features.index('centroid_head_y')]
            chix1 = track['data'][0, :, features.index('centroid_hips_x')]
            chiy1 = track['data'][0, :, features.index('centroid_hips_y')]
            chx2 = track['data'][1, :, features.index('centroid_head_x')]
            chy2 = track['data'][1, :, features.index('centroid_head_y')]
            chix2 = track['data'][1, :, features.index('centroid_hips_x')]
            chiy2 = track['data'][1, :, features.index('centroid_hips_y')]
            x_acc = np.diff(np.vstack((chx1, chix1)), n=2)
            y_acc = np.diff(np.vstack((chy1, chiy1)), n=2)
            acc1 = np.linalg.norm([x_acc, y_acc], axis=(0, 1))
            track['data'][0, :, ind] = np.hstack((np.hstack((acc1[0], acc1)), acc1[-1])) * fps / (pix_per_cm ** 2)
            x_acc = np.diff(np.vstack((chx2, chix2)), n=2)
            y_acc = np.diff(np.vstack((chy2, chiy2)), n=2)
            acc2 = np.linalg.norm([x_acc, y_acc], axis=(0, 1))
            track['data'][1, :, ind] = np.hstack((np.hstack((acc2[0], acc2)), acc2[-1])) * fps / (pix_per_cm ** 2)

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            chix1[1:-1] = sig.convolve(chix1, smooth_kernel, 'valid')
            chiy1[1:-1] = sig.convolve(chiy1, smooth_kernel, 'valid')
            chix2[1:-1] = sig.convolve(chix2, smooth_kernel, 'valid')
            chiy2[1:-1] = sig.convolve(chiy2, smooth_kernel, 'valid')
            x_acc = np.diff(np.vstack((chx1, chix1)), n=2)
            y_acc = np.diff(np.vstack((chy1, chiy1)), n=2)
            acc1 = np.linalg.norm([x_acc, y_acc], axis=(0, 1))
            track['data_smooth'][0, :, ind] = np.hstack((np.hstack((acc1[0], acc1)), acc1[-1])) * fps / (pix_per_cm ** 2)
            x_acc = np.diff(np.vstack((chx2, chix2)), n=2)
            y_acc = np.diff(np.vstack((chy2, chiy2)), n=2)
            acc2 = np.linalg.norm([x_acc, y_acc], axis=(0, 1))
            track['data_smooth'][1, :, ind] = np.hstack((np.hstack((acc2[0], acc2)), acc2[-1])) * fps / (pix_per_cm ** 2)

        # distance between mice's centroids
        if 'rel_dist_centroid' in features:
            ind = features.index('rel_dist_centroid')
            cent_x_ind = features.index('centroid_x')
            cent_y_ind = features.index('centroid_y')
            x_dif = track['data'][0, :, cent_x_ind] - track['data'][1, :, cent_x_ind]
            y_dif = track['data'][0, :, cent_y_ind] - track['data'][1, :, cent_y_ind]
            cent_dif = np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)
            track['data'][0, :, ind] = cent_dif / pix_per_cm
            track['data'][1, :, ind] = cent_dif / pix_per_cm
            cent_dif[1:-1] = sig.convolve(cent_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = cent_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = cent_dif / pix_per_cm

        # distance change mice centroids
        if 'rel_dist_centroid_change' in features:
            ind = features.index('rel_dist_centroid_change')
            dist_bet = np.diff(track['data'][0, :, features.index('rel_dist_centroid')] * pix_per_cm)
            rdcc = np.hstack((dist_bet[0], dist_bet))
            track['data'][0, :, ind] = rdcc / pix_per_cm
            track['data'][1, :, ind] = rdcc / pix_per_cm
            rdcc[1:-1] = sig.convolve(rdcc, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = rdcc / pix_per_cm
            track['data_smooth'][1, :, ind] = rdcc / pix_per_cm

        # distance between mice's nose
        if 'rel_dist_nose' in features:
            ind = features.index('rel_dist_nose')
            nose_x_ind = features.index('nose_x')
            nose_y_ind = features.index('nose_y')
            x_dif = track['data'][1, :, nose_x_ind] - track['data'][0, :, nose_x_ind]
            y_dif = track['data'][1, :, nose_y_ind] - track['data'][0, :, nose_y_ind]
            nose_dif = np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)
            track['data'][0, :, ind] = nose_dif / pix_per_cm
            track['data'][1, :, ind] = nose_dif / pix_per_cm
            nose_dif[1:-1] = sig.convolve(nose_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = nose_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = nose_dif / pix_per_cm

        # rel dist mice's head
        if 'rel_dist_head' in features:
            ind = features.index('rel_dist_head')  # centroid_body_x
            h_x_ind = features.index('centroid_head_x')
            h_y_ind = features.index('centroid_head_y')
            x_dif = track['data'][1, :, h_x_ind] - track['data'][0, :, h_x_ind]
            y_dif = track['data'][1, :, h_y_ind] - track['data'][0, :, h_y_ind]
            bod_dif = np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)
            track['data'][0, :, ind] = bod_dif / pix_per_cm
            track['data'][1, :, ind] = bod_dif / pix_per_cm
            bod_dif[1:-1] = sig.convolve(bod_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = bod_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = bod_dif / pix_per_cm

        # distance between mice's body centroids
        if 'rel_dist_body' in features:
            ind = features.index('rel_dist_body')  # centroid_body_x
            bod_x_ind = features.index('centroid_body_x')
            bod_y_ind = features.index('centroid_body_y')
            x_dif = track['data'][1, :, bod_x_ind] - track['data'][0, :, bod_x_ind]
            y_dif = track['data'][1, :, bod_y_ind] - track['data'][0, :, bod_y_ind]
            bod_dif = np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)
            track['data'][0, :, ind] = bod_dif / pix_per_cm
            track['data'][1, :, ind] = bod_dif / pix_per_cm
            bod_dif[1:-1] = sig.convolve(bod_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = bod_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = bod_dif / pix_per_cm

        # distance between mice's head body
        if 'rel_dist_head_body' in features:
            ind = features.index('rel_dist_head_body')  # centroid_body_x
            bod_x_ind = features.index('centroid_body_x')
            bod_y_ind = features.index('centroid_body_y')
            h_x_ind = features.index('centroid_head_x')
            h_y_ind = features.index('centroid_head_y')
            x1_dif = track['data'][0, :, h_x_ind] - track['data'][1, :, bod_x_ind]
            y1_dif = track['data'][0, :, h_y_ind] - track['data'][1, :, bod_y_ind]
            x2_dif = track['data'][1, :, h_x_ind] - track['data'][0, :, bod_x_ind]
            y2_dif = track['data'][1, :, h_y_ind] - track['data'][0, :, bod_y_ind]
            hb1_dif = np.linalg.norm(np.vstack((x1_dif, y1_dif)), axis=0)
            hb2_dif = np.linalg.norm(np.vstack((x2_dif, y2_dif)), axis=0)
            track['data'][0, :, ind] = hb1_dif / pix_per_cm
            track['data'][1, :, ind] = hb2_dif / pix_per_cm
            hb1_dif[1:-1] = sig.convolve(hb1_dif, smooth_kernel, 'valid')
            hb2_dif[1:-1] = sig.convolve(hb2_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = hb1_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = hb2_dif / pix_per_cm

        # Angle of heading relative to other animal (PNAS Feat. 12, Eyrun's Feat. 11)
        if 'rel_angle_social' in features:
            ind = features.index('rel_angle_social')
            bod_x_ind = features.index('centroid_x')
            bod_y_ind = features.index('centroid_y')
            ang_ind = features.index('ori_body')
            x_dif = track['data'][1, :, bod_x_ind] - track['data'][0, :, bod_x_ind]  # r_x
            y_dif = track['data'][1, :, bod_y_ind] - track['data'][0, :, bod_y_ind]  # r_y
            theta0 = (np.arctan2(y_dif, x_dif) + 2 * np.pi) % 2 * np.pi
            theta1 = (np.arctan2(-y_dif, -x_dif) + 2 * np.pi) % 2 * np.pi
            ang0 = np.mod(theta0 - track['data'][0, :, ang_ind], 2 * np.pi)
            ang1 = np.mod(theta1 - track['data'][1, :, ang_ind], 2 * np.pi)
            track['data'][0, :, ind] = np.minimum(ang0, 2 * np.pi - ang0)
            track['data'][1, :, ind] = np.minimum(ang1, 2 * np.pi - ang1)
            track['data_smooth'][0, :, ind] = track['data'][0, :, ind]
            track['data_smooth'][0, 1:-1, ind] = np.mod(sig.convolve(track['data_smooth'][0, :, ind], smooth_kernel, 'valid') + 2 * np.pi, 2 * np.pi)
            track['data_smooth'][1, :, ind] = track['data'][1, :, ind]
            track['data_smooth'][1, 1:-1, ind] = np.mod(sig.convolve(track['data_smooth'][1, :, ind], smooth_kernel, 'valid') + 2 * np.pi, 2 * np.pi)

        # semiaxis length
        M_ind = features.index('major_axis_len')
        m_ind = features.index('minor_axis_len')
        phi_ind = features.index('phi')
        c_M_0 = np.multiply(track['data'][0, :, M_ind] * pix_per_cm, np.sin(track['data'][0, :, phi_ind]))
        c_m_0 = np.multiply(track['data'][0, :, m_ind] * pix_per_cm, np.cos(track['data'][0, :, phi_ind]))
        c_M_1 = np.multiply(track['data'][1, :, M_ind] * pix_per_cm, np.sin(track['data'][1, :, phi_ind]))
        c_m_1 = np.multiply(track['data'][1, :, m_ind] * pix_per_cm, np.cos(track['data'][1, :, phi_ind]))

        # distance as defined in PNAS (Feat. 13)
        if 'rel_dist_gap' in features:
            ind = features.index('rel_dist_gap')
            rel_dist_ind = features.index('rel_dist_body')
            comb_norm = np.linalg.norm(np.vstack((c_M_0, c_m_0)), axis=0) + np.linalg.norm(np.vstack((c_M_1, c_m_1)), axis=0)
            d1 = (track['data'][0, :, rel_dist_ind] * pix_per_cm - comb_norm)
            d2 = (track['data'][1, :, rel_dist_ind] * pix_per_cm - comb_norm)
            track['data'][0, :, ind] = d1 / pix_per_cm
            track['data'][1, :, ind] = d2 / pix_per_cm
            d1[1:-1] = sig.convolve(d1, smooth_kernel, 'valid')
            d2[1:-1] = sig.convolve(d2, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = d1 / pix_per_cm
            track['data_smooth'][1, :, ind] = d2 / pix_per_cm

        # distance in units of semiaxis (Feat. 14)
        if 'rel_dist_scaled' in features:
            ind = features.index('rel_dist_scaled')
            rel_dist_gap_ind = features.index('rel_dist_gap')
            d1 = np.divide(track['data'][0, :, rel_dist_gap_ind] * pix_per_cm, np.linalg.norm(np.vstack((c_M_0, c_m_0)), axis=0))
            d2 = np.divide(track['data'][1, :, rel_dist_gap_ind] * pix_per_cm, np.linalg.norm(np.vstack((c_M_1, c_m_1)), axis=0))
            track['data'][0, :, ind] = d1 / pix_per_cm
            track['data'][1, :, ind] = d2 / pix_per_cm
            d1[1:-1] = sig.convolve(d1, smooth_kernel, 'valid')
            d2[1:-1] = sig.convolve(d2, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = d1 / pix_per_cm
            track['data_smooth'][1, :, ind] = d2 / pix_per_cm

        # Ratio of mouse 1's ellipse area to mouse 0's ellipse area (Feat. 15)
        if 'area_ellipse_ratio' in features:
            ind = features.index('area_ellipse_ratio')
            area_ind = features.index('area_ellipse')
            track['data'][0, :, ind] = np.divide(track['data'][0, :, area_ind], track['data'][1, :, area_ind], dtype=float)
            track['data'][1, :, ind] = np.divide(track['data'][1, :, area_ind], track['data'][0, :, area_ind], dtype=float)
            track['data_smooth'][0, :, ind] = track['data'][0, :, ind]
            track['data_smooth'][0, 1:-1, ind] = sig.convolve(track['data_smooth'][0, :, ind], smooth_kernel, 'valid')
            track['data_smooth'][1, :, ind] = track['data'][1, :, ind]
            track['data_smooth'][1, 1:-1, ind] = sig.convolve(track['data_smooth'][1, :, ind], smooth_kernel, 'valid')

        # angle between & facing angle
        if 'angle_between' in features:
            ind = features.index('angle_between')
            ori_ind = features.index(('phi'))
            vec_rot = np.vstack((np.cos(track['data'][0, :, ori_ind]), -np.sin(track['data'][0, :, ori_ind])))
            vec_rot2 = np.vstack((np.cos(track['data'][1, :, ori_ind]), -np.sin(track['data'][1, :, ori_ind])))
            angle_btw = np.arccos((vec_rot * vec_rot2).sum(axis=0))
            track['data'][0, :, ind] = angle_btw
            track['data'][1, :, ind] = angle_btw
            angle_btw[1:-1] = sig.convolve(angle_btw, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = angle_btw
            track['data_smooth'][1, :, ind] = angle_btw

        if 'facing_angle' in features:
            ori_ind = features.index(('phi'))
            vec_rot = np.vstack((np.cos(track['data'][0, :, ori_ind]), -np.sin(track['data'][0, :, ori_ind])))
            vec_rot2 = np.vstack((np.cos(track['data'][1, :, ori_ind]), -np.sin(track['data'][1, :, ori_ind])))
            c1 = np.vstack((track['data'][0, :, features.index('centroid_x')], track['data'][0, :, features.index('centroid_y')]))
            c2 = np.vstack((track['data'][1, :, features.index('centroid_x')], track['data'][1, :, features.index('centroid_y')]))
            vec_btw = c2 - c1
            norm_btw = np.linalg.norm(np.vstack((vec_btw[0, :], vec_btw[1, :])), axis=0)
            vec_btw = vec_btw / np.repeat([norm_btw], 2, axis=0)
            fa1 = np.arccos((vec_rot * vec_btw).sum(axis=0))
            fa2 = np.arccos((vec_rot2 * -vec_btw).sum(axis=0))
            ind = features.index('facing_angle')
            track['data'][0, :, ind] = fa1
            track['data'][1, :, ind] = fa2
            fa1[1:-1] = sig.convolve(fa1, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = fa1
            fa2[1:-1] = sig.convolve(fa2, smooth_kernel, 'valid')
            track['data_smooth'][1, :, ind] = fa2

        # distance points mouse1 to mouse 2, m1 and m1 , m2 and m2
        if 'dist_m1nose_m2nose' in features or 'dist_nose_right_ear' in features:
            if 'dist_m1nose_m2nose' in features:
                m1m2d = np.zeros((num_frames, num_points ** 2))
                idx_triu = np.triu_indices(num_points, 1)
                n_idx = len(idx_triu[0])
            if 'dist_nose_right_ear' in features:
                m1m1d = np.zeros((num_frames, n_idx))
                m2m2d = np.zeros((num_frames, n_idx))

            for f in range(num_frames):
                m1 = track['data'][0, f, :(num_points * 2)]
                m2 = track['data'][1, f, :(num_points * 2)]
                m1_x = m1[::2]
                m1_y = m1[1::2]
                m2_x = m2[::2]
                m2_y = m2[1::2]
                m1_p = np.vstack((m1_x, m1_y)).T
                m2_p = np.vstack((m2_x, m2_y)).T

                if 'dist_m1nose_m2nose' in features:
                    m1m2d[f, :] = dist.cdist(m1_p, m2_p).flatten()
                if 'dist_nose_right_ear' in features:
                    m1m1d[f, :] = dist.cdist(m1_p, m1_p)[idx_triu]
                    m2m2d[f, :] = dist.cdist(m2_p, m2_p)[idx_triu]

            if 'dist_m1nose_m2nose' in features:
                ind_m1m2d = features.index('dist_m1nose_m2nose')
                track['data'][0, :, ind_m1m2d:ind_m1m2d + num_points ** 2] = m1m2d / pix_per_cm
                track['data'][1, :, ind_m1m2d:ind_m1m2d + num_points ** 2] = m1m2d / pix_per_cm
                for d in range(m1m2d.shape[1]): m1m2d[1:-1, d] = sig.convolve(m1m2d[:, d], smooth_kernel, 'valid')
                track['data_smooth'][0, :, ind_m1m2d:ind_m1m2d + num_points ** 2] = m1m2d / pix_per_cm
                track['data_smooth'][1, :, ind_m1m2d:ind_m1m2d + num_points ** 2] = m1m2d / pix_per_cm
            if 'dist_nose_right_ear' in features:
                ind_md = features.index('dist_nose_right_ear')
                track['data'][0, :, ind_md:ind_md + n_idx] = m1m1d / pix_per_cm
                track['data'][1, :, ind_md:ind_md + n_idx] = m2m2d / pix_per_cm
                for d in range(m1m1d.shape[1]): m1m1d[1:-1, d] = sig.convolve(m1m1d[:, d], smooth_kernel, 'valid')
                for d in range(m2m2d.shape[1]): m2m2d[1:-1, d] = sig.convolve(m2m2d[:, d], smooth_kernel, 'valid')
                track['data_smooth'][0, :, ind_md:ind_md + n_idx] = m1m1d / pix_per_cm
                track['data_smooth'][1, :, ind_md:ind_md + n_idx] = m2m2d / pix_per_cm

        # velocity centroid windows -2 -5 -10
        if 'speed_centroid' in features or 'speed_centroid_x' in features or 'speed_centroid_y' in features:
            chx1 = track['data'][0, :, features.index('centroid_x')]
            chy1 = track['data'][0, :, features.index('centroid_y')]
            chx2 = track['data'][1, :, features.index('centroid_x')]
            chy2 = track['data'][1, :, features.index('centroid_y')]
            x_diff0 = np.diff(chx1)
            y_diff0 = np.diff(chy1)
            x_diff1 = np.diff(chx2)
            y_diff1 = np.diff(chy2)
            if 'speed_centroid' in features:
                ind = features.index('speed_centroid')
                vel0 = np.linalg.norm([x_diff0, y_diff0], axis=0)
                track['data'][0, :, ind] = np.hstack((vel0[0], vel0)) * fps / pix_per_cm
                vel1 = np.linalg.norm([x_diff1, y_diff1], axis=0)
                track['data'][1, :, ind] = np.hstack((vel1[0], vel1)) * fps / pix_per_cm

            if 'speed_centroid_x' in features:
                indx = features.index('speed_centroid_x')
                track['data'][0, :, indx] = np.hstack((x_diff0[0], x_diff0))
                track['data'][1, :, indx] = np.hstack((x_diff1[0], x_diff1))

            if 'speed_centroid_y' in features:
                indy = features.index('speed_centroid_y')
                track['data'][0, :, indy] = np.hstack((y_diff0[0], y_diff0))
                track['data'][1, :, indy] = np.hstack((y_diff1[0], y_diff1))

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            x_diff0 = np.diff(chx1)
            y_diff0 = np.diff(chy1)
            x_diff1 = np.diff(chx2)
            y_diff1 = np.diff(chy2)

            if 'speed_centroid_x' in features:
                track['data_smooth'][0, :, indx] = np.hstack((x_diff0[0], x_diff0))
                track['data_smooth'][1, :, indx] = np.hstack((x_diff1[0], x_diff1))
            if 'speed_centroid_y' in features:
                track['data_smooth'][0, :, indy] = np.hstack((y_diff0[0], y_diff0))
                track['data_smooth'][1, :, indy] = np.hstack((y_diff1[0], y_diff1))
            if 'speed_centroid' in features:
                vel0 = np.linalg.norm([x_diff0, y_diff0], axis=0)
                track['data_smooth'][0, :, ind] = np.hstack((vel0[0], vel0)) * fps / pix_per_cm
                vel1 = np.linalg.norm([x_diff1, y_diff1], axis=0)
                track['data_smooth'][1, :, ind] = np.hstack((vel1[0], vel1)) * fps / pix_per_cm

        # acceleration centroid
        if 'acceleration_centroid' in features or 'acceleration_centroid_x' in features or 'acceleration_centroid_y' in features:
            chx1 = track['data'][0, :, features.index('centroid_x')]
            chy1 = track['data'][0, :, features.index('centroid_y')]
            chx2 = track['data'][1, :, features.index('centroid_x')]
            chy2 = track['data'][1, :, features.index('centroid_y')]
            x_acc0 = np.diff(chx1, n=2)
            y_acc0 = np.diff(chy1, n=2)
            x_acc1 = np.diff(chx2, n=2)
            y_acc1 = np.diff(chy2, n=2)

            if 'acceleration_centroid' in features:
                ind = features.index('acceleration_centroid')
                acc0 = np.linalg.norm([x_acc0, y_acc0], axis=0)
                acc1 = np.linalg.norm([x_acc1, y_acc1], axis=0)
                track['data'][0, :, ind] = np.hstack((acc0[0], acc0[0], acc0)) * fps / (pix_per_cm ** 2)
                track['data'][1, :, ind] = np.hstack((acc1[0], acc1[0], acc1)) * fps / (pix_per_cm ** 2)

            if 'acceleration_centroid_x' in features:
                indx = features.index('acceleration_centroid_x')
                track['data'][0, :, indx] = np.hstack((x_acc0[0], x_acc0[0], x_acc0))
                track['data'][1, :, indx] = np.hstack((x_acc1[0], x_acc1[0], x_acc1))

            if 'acceleration_centroid_y' in features:
                indy = features.index('acceleration_centroid_y')
                track['data'][0, :, indy] = np.hstack((y_acc0[0], y_acc0[0], y_acc0))
                track['data'][1, :, indy] = np.hstack((y_acc1[0], y_acc1[0], y_acc1))

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            x_acc0 = np.diff(chx1, n=2)
            y_acc0 = np.diff(chy1, n=2)
            x_acc1 = np.diff(chx2, n=2)
            y_acc1 = np.diff(chy2, n=2)

            if 'acceleration_centroid' in features:
                acc0 = np.linalg.norm([x_acc0, y_acc0], axis=0)
                acc1 = np.linalg.norm([x_acc1, y_acc1], axis=0)
                track['data_smooth'][0, :, ind] = np.hstack((acc0[0], acc0[0], acc0)) * fps / (pix_per_cm ** 2)
                track['data_smooth'][1, :, ind] = np.hstack((acc1[0], acc1[0], acc1)) * fps / (pix_per_cm ** 2)

            if 'acceleration_centroid_x' in features:
                track['data_smooth'][0, :, indx] = np.hstack((x_acc0[0], x_acc0[0], x_acc0))
                track['data_smooth'][1, :, indx] = np.hstack((x_acc1[0], x_acc1[0], x_acc1))

            if 'acceleration_centroid_y' in features:
                track['data_smooth'][0, :, indy] = np.hstack((y_acc0[0], y_acc0[0], y_acc0))
                track['data_smooth'][1, :, indy] = np.hstack((y_acc1[0], y_acc1[0], y_acc1))

        if 'speed_centroid_w2' in features:
            ind_w2 = features.index('speed_centroid_w2')
            ind_w5 = features.index('speed_centroid_w5')
            ind_w10 = features.index('speed_centroid_w10')
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(2, num_frames - 2):
                x_diff0 = chx1[f] - chx1[f - 2]
                y_diff0 = chy1[f] - chy1[f - 2]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 2]
                y_diff1 = chy2[f] - chy2[f - 2]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data'][0, :, ind_w2] = vel1 * fps / pix_per_cm
            track['data'][1, :, ind_w2] = vel2 * fps / pix_per_cm
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(5, num_frames - 5):
                x_diff0 = chx1[f] - chx1[f - 5]
                y_diff0 = chy1[f] - chy1[f - 5]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 5]
                y_diff1 = chy2[f] - chy2[f - 5]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data'][0, :, ind_w5] = vel1 * fps / pix_per_cm
            track['data'][1, :, ind_w5] = vel2 * fps / pix_per_cm
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(10, num_frames - 10):
                x_diff0 = chx1[f] - chx1[f - 10]
                y_diff0 = chy1[f] - chy1[f - 10]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 10]
                y_diff1 = chy2[f] - chy2[f - 10]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data'][0, :, ind_w10] = vel1 * fps / pix_per_cm
            track['data'][1, :, ind_w10] = vel2 * fps / pix_per_cm

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(2, num_frames - 2):
                x_diff0 = chx1[f] - chx1[f - 2]
                y_diff0 = chy1[f] - chy1[f - 2]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 2]
                y_diff1 = chy2[f] - chy2[f - 2]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data_smooth'][0, :, ind_w2] = vel1 * fps / pix_per_cm
            track['data_smooth'][1, :, ind_w2] = vel2 * fps / pix_per_cm
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(5, num_frames - 5):
                x_diff0 = chx1[f] - chx1[f - 5]
                y_diff0 = chy1[f] - chy1[f - 5]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 5]
                y_diff1 = chy2[f] - chy2[f - 5]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data_smooth'][0, :, ind_w5] = vel1 * fps / pix_per_cm
            track['data_smooth'][1, :, ind_w5] = vel2 * fps / pix_per_cm
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(10, num_frames - 10):
                x_diff0 = chx1[f] - chx1[f - 10]
                y_diff0 = chy1[f] - chy1[f - 10]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 10]
                y_diff1 = chy2[f] - chy2[f - 10]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data_smooth'][0, :, ind_w10] = vel1 * fps / pix_per_cm
            track['data_smooth'][1, :, ind_w10] = vel2 * fps / pix_per_cm

        # velocity points windows -2 -5 -10
        if 'speed_nose_w2' in features:
            for p in range(0, num_points * 2, 2):
                chx1 = track['data'][0, :, p]
                chy1 = track['data'][0, :, p + 1]
                chx2 = track['data'][1, :, p]
                chy2 = track['data'][1, :, p + 1]
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                ind_w2 = features.index('speed_' + parts[int(p / 2)] + '_w2')
                ind_w5 = features.index('speed_' + parts[int(p / 2)] + '_w5')
                ind_w10 = features.index('speed_' + parts[int(p / 2)] + '_w10')
                for f in range(2, num_frames - 2):
                    x_diff0 = chx1[f] - chx1[f - 2]
                    y_diff0 = chy1[f] - chy1[f - 2]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 2]
                    y_diff1 = chy2[f] - chy2[f - 2]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data'][0, :, ind_w2] = vel1 * fps / pix_per_cm
                track['data'][1, :, ind_w2] = vel2 * fps / pix_per_cm
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(5, num_frames - 5):
                    x_diff0 = chx1[f] - chx1[f - 5]
                    y_diff0 = chy1[f] - chy1[f - 5]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 5]
                    y_diff1 = chy2[f] - chy2[f - 5]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data'][0, :, ind_w5] = vel1 * fps / pix_per_cm
                track['data'][1, :, ind_w5] = vel2 * fps / pix_per_cm
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(10, num_frames - 10):
                    x_diff0 = chx1[f] - chx1[f - 10]
                    y_diff0 = chy1[f] - chy1[f - 10]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 10]
                    y_diff1 = chy2[f] - chy2[f - 10]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data'][0, :, ind_w10] = vel1 * fps / pix_per_cm
                track['data'][1, :, ind_w10] = vel2 * fps / pix_per_cm

                chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
                chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
                chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
                chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(2, num_frames - 2):
                    x_diff0 = chx1[f] - chx1[f - 2]
                    y_diff0 = chy1[f] - chy1[f - 2]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 2]
                    y_diff1 = chy2[f] - chy2[f - 2]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data_smooth'][0, :, ind_w2] = vel1 * fps / pix_per_cm
                track['data_smooth'][1, :, ind_w2] = vel2 * fps / pix_per_cm
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(5, num_frames - 5):
                    x_diff0 = chx1[f] - chx1[f - 5]
                    y_diff0 = chy1[f] - chy1[f - 5]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 5]
                    y_diff1 = chy2[f] - chy2[f - 5]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data_smooth'][0, :, ind_w5] = vel1 * fps / pix_per_cm
                track['data_smooth'][1, :, ind_w5] = vel2 * fps / pix_per_cm
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(10, num_frames - 10):
                    x_diff0 = chx1[f] - chx1[f - 10]
                    y_diff0 = chy1[f] - chy1[f - 10]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 10]
                    y_diff1 = chy2[f] - chy2[f - 10]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data_smooth'][0, :, ind_w10] = vel1 * fps / pix_per_cm
                track['data_smooth'][1, :, ind_w10] = vel2 * fps / pix_per_cm

        # distance to edge
        if 'dist_edge_x' in features:
            ind = features.index('dist_edge_x')
            allx = np.concatenate(allx).ravel()
            ally = np.concatenate(ally).ravel()
            allx = allx[allx > 0]
            ally = ally[ally > 0]
            minx = np.percentile(allx, 1)
            maxx = np.percentile(allx, 99)
            miny = np.percentile(ally, 1)
            maxy = np.percentile(ally, 99)

            cx1 = track['data'][0, :, features.index('centroid_x')]
            cx2 = track['data'][1, :, features.index('centroid_x')]
            cy1 = track['data'][0, :, features.index('centroid_y')]
            cy2 = track['data'][1, :, features.index('centroid_y')]
            distsx1 = np.amin(np.stack((np.maximum(0, cx1 - minx), np.maximum(0, maxx - cx1)), axis=-1), axis=1)
            distsx2 = np.amin(np.stack((np.maximum(0, cx2 - minx), np.maximum(0, maxx - cx2)), axis=-1), axis=1)
            distsy1 = np.amin(np.stack((np.maximum(0, cy1 - miny), np.maximum(0, maxy - cy1)), axis=-1), axis=1)
            distsy2 = np.amin(np.stack((np.maximum(0, cy2 - miny), np.maximum(0, maxy - cy2)), axis=-1), axis=1)
            dists1 = np.amin(np.stack((distsx1, distsy1), axis=-1), axis=1)
            dists2 = np.amin(np.stack((distsx2, distsy2), axis=-1), axis=1)
            track['data'][0, :, ind] = distsx1
            track['data'][0, :, ind + 1] = distsy1
            track['data'][0, :, ind + 2] = dists1
            track['data'][1, :, ind] = distsx2
            track['data'][1, :, ind + 1] = distsy2
            track['data'][1, :, ind + 2] = dists2
            # distsx1[1:-1]= sig.convolve(distsx1, smooth_kernel, 'valid')
            # distsx2[1:-1]= sig.convolve(distsx2, smooth_kernel, 'valid')
            # distsy1[1:-1]= sig.convolve(distsy1, smooth_kernel, 'valid')
            # distsy2[1:-1]= sig.convolve(distsy2, smooth_kernel, 'valid')
            # dists1[1:-1]= sig.convolve(dists1, smooth_kernel, 'valid')
            # dists2[1:-1]= sig.convolve(dists2, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = distsx1
            track['data_smooth'][0, :, ind + 1] = distsy1
            track['data_smooth'][0, :, ind + 2] = dists1
            track['data_smooth'][1, :, ind] = distsx2
            track['data_smooth'][1, :, ind + 1] = distsy2
            track['data_smooth'][1, :, ind + 2] = dists2

        # radial vel and tangetial vel
        if 'radial_vel' in features or ' tangential_vel' in features:
            cx1 = track['data'][0, :, features.index('centroid_x')]
            cx2 = track['data'][1, :, features.index('centroid_x')]
            cy1 = track['data'][0, :, features.index('centroid_y')]
            cy2 = track['data'][1, :, features.index('centroid_y')]
            ddx1 = cx2 - cx1
            ddx2 = cx1 - cx2
            ddy1 = cy2 - cy1
            ddy2 = cy1 - cy2
            len1 = np.sqrt(ddx1 ** 2 + ddy1 ** 2)
            len1[len1 == 0] = eps
            len2 = np.sqrt(ddx2 ** 2 + ddy2 ** 2)
            len2[len2 == 0] = eps
            ddx1 = ddx1 / len1
            ddx2 = ddx2 / len2
            ddy1 = ddy1 / len1
            ddy2 = ddy2 / len2
            dx1 = np.diff(cx1)
            dx2 = np.diff(cx2)
            dy1 = np.diff(cy1)
            dy2 = np.diff(cy2)

            if 'radial_vel' in features:
                ind = features.index('radial_vel')
                radial_vel1 = dx1 * ddx1[1:] + dy1 * ddy1[1:]
                radial_vel2 = dx2 * ddx2[1:] + dy2 * ddy2[1:]
                track['data'][0, :, ind] = np.hstack((radial_vel1[0], radial_vel1))
                track['data'][1, :, ind] = np.hstack((radial_vel2[0], radial_vel2))
                radial_vel1[1:-1] = sig.convolve(radial_vel1, smooth_kernel, 'valid')
                radial_vel2[1:-1] = sig.convolve(radial_vel2, smooth_kernel, 'valid')
                track['data_smooth'][0, :, ind] = np.hstack((radial_vel1[0], radial_vel1))
                track['data_smooth'][1, :, ind] = np.hstack((radial_vel2[0], radial_vel2))

            # tangential vel
            if 'tangential_vel' in features:
                ind = features.index('tangential_vel')
                orth_ddx1 = -ddy1
                orth_ddx2 = -ddy2
                orth_ddy1 = -ddx1
                orth_ddy2 = -ddx2
                tang_vel1 = np.abs(dx1 * orth_ddx1[1:] + dy1 * orth_ddy1[1:])
                tang_vel2 = np.abs(dx2 * orth_ddx2[1:] + dy2 * orth_ddy2[1:])
                track['data'][0, :, ind] = np.hstack((tang_vel1[0], tang_vel1))
                track['data'][1, :, ind] = np.hstack((tang_vel2[0], tang_vel2))
                tang_vel1[1:-1] = sig.convolve(tang_vel1, smooth_kernel, 'valid')
                tang_vel2[1:-1] = sig.convolve(tang_vel2, smooth_kernel, 'valid')
                track['data_smooth'][0, :, ind] = np.hstack((tang_vel1[0], tang_vel1))
                track['data_smooth'][1, :, ind] = np.hstack((tang_vel2[0], tang_vel2))


                ################################ pixel based features

        # pixel based features pnas
        if 'imgDf1' in features:
            def smooth(a, WSZ):
                out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
                r = np.arange(1, WSZ - 1, 2)
                start = np.cumsum(a[:WSZ - 1])[::2] / r
                stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
                return np.concatenate((start, out0, stop))

            blur = np.floor(fps * 1.5)
            ind1 = features.index('imgDf1')
            ind7 = features.index('imgDf1Smooth')
            ind2 = features.index('imgDf2')
            ind3 = features.index('imgDf3')
            ind4 = features.index('imgDf6')
            ind5 = features.index('imgMean')
            ind6 = features.index('imgStd')

            for f in range(num_frames):
                fFr = syncTopFront(f,num_frames,num_framesf) if flag_align else f
                fr = readerf.getFrame(fFr)
                fr = fr.astype(np.float32)

                track['data_smooth'][0, f, ind5] = np.mean(fr)
                track['data_smooth'][0, f, ind6] = np.std(fr)


            pr = np.linspace(0,num_frames/2,num_frames)
            for f in range(1, num_frames):
                bar.update(pr[f])

                if progress_bar_sig:
                    if pr[f] <= 1:
                        progress_bar_sig.emit(pr[f],num_frames-1)
                    progress_bar_sig.emit(pr[f],0)

                fFr = syncTopFront(f,num_frames,num_framesf) if flag_align else f
                imgMean = track['data_smooth'][0, f - 1, ind5]
                imgStd = track['data_smooth'][0, f - 1, ind6]
                if f == 1:
                    frame1 = readerf.getFrame(fFr - 1)
                    frame1 = frame1.astype(np.float32)
                else:
                    frame1 = frame2

                frame2 = readerf.getFrame(fFr)
                frame2 = frame2.astype(np.float32)

                df = np.abs(frame2 - frame1)
                dfm = np.mean(df * df)
                track['data_smooth'][0, f, ind1] = dfm / (imgMean ** 2)
                track['data_smooth'][0, f, ind2] = (np.mean(df) / imgMean) ** 2
                track['data_smooth'][0, f, ind3] = (np.mean(df / (frame1 + eps))) ** 2
                track['data_smooth'][0, f, ind4] = dfm / (imgStd ** 2)

            track['data_smooth'][0, :, ind7] = smooth(track['data_smooth'][0, :, ind1], int(blur * 2 + 1)) * 1200
            track['data_smooth'][1, :, ind1] = track['data_smooth'][0, :, ind1]
            track['data_smooth'][1, :, ind2] = track['data_smooth'][0, :, ind2]
            track['data_smooth'][1, :, ind3] = track['data_smooth'][0, :, ind3]
            track['data_smooth'][1, :, ind4] = track['data_smooth'][0, :, ind4]
            track['data_smooth'][1, :, ind5] = track['data_smooth'][0, :, ind5]
            track['data_smooth'][1, :, ind6] = track['data_smooth'][0, :, ind6]
            track['data_smooth'][1, :, ind7] = track['data_smooth'][0, :, ind7]

        # pixel based features
        if 'pixel_change' in features or 'pixel_change_ubbox_mice' in features or 'pixel_change_front' in features or 'nose_pc' in features:

            if 'pixel_change' in features:
                ind = features.index('pixel_change')
            if 'pixel_change_ubbox_mice' in features:
                ind1 = features.index('pixel_change_ubbox_mice')
            if 'pixel_change_front' in features:
                indf = features.index('pixel_change_front')
            if 'nose_pc' in features:
                indn = features.index('nose_pc')

            pr = np.linspace(num_frames/2,num_frames-1,num_frames)
            for f in range(1, num_frames):
                bar.update(pr[f])

                if progress_bar_sig:
                    if pr[f] <= 1:
                        progress_bar_sig.emit(pr[f],num_frames-1)
                    progress_bar_sig.emit(pr[f],0)

                fFr = syncTopFront(f,num_frames,num_framesf) if flag_align else f
                if f == 1:
                    frame1 = reader.getFrame(f - 1)
                    frame1 = frame1.astype(np.float32)

                    frame1f = readerf.getFrame(fFr - 1)
                    frame1f = frame1f.astype(np.float32)
                    frame1u = frame1
                else:
                    frame1 = frame2
                    frame1f = frame2f
                    frame1u = frame2

                frame2 = reader.getFrame(f)
                frame2 = frame2.astype(np.float32)

                frame2f = readerf.getFrame(fFr)
                frame2f = frame2f.astype(np.float32)
                frame2u = frame2


                if 'pixel_change' in features:
                    track['data'][0, f, ind] = (np.sum((frame2 - frame1) ** 2)) / float((np.sum((frame1) ** 2)))
                    track['data'][1, f, ind] = track['data'][0, f, ind]

                if 'pixel_change_ubbox_mice' in features:
                    ind_bbo = features.index('overlap_bboxes')
                    f1_bb = track['data'][0, f - 1, ind_bbo]
                    f2_bb = track['data'][0, f, ind_bbo]

                    if f1_bb > 0. or f2_bb > 0.:
                        bbox_f1_m1 = track['bbox'][0, :, f - 1]
                        xmin11, xmax11 = bbox_f1_m1[[0, 2]] * im_w
                        ymin11, ymax11 = bbox_f1_m1[[1, 3]] * im_h
                        bbox_f1_m2 = track['bbox'][1, :, f - 1]
                        xmin12, xmax12 = bbox_f1_m2[[0, 2]] * im_w
                        ymin12, ymax12 = bbox_f1_m2[[1, 3]] * im_h

                        bbox_f2_m1 = track['bbox'][0, :, f]
                        xmin21, xmax21 = bbox_f2_m1[[0, 2]] * im_w
                        ymin21, ymax21 = bbox_f2_m1[[1, 3]] * im_h
                        bbox_f2_m2 = track['bbox'][1, :, f]
                        xmin22, xmax22 = bbox_f2_m2[[0, 2]] * im_w
                        ymin22, ymax22 = bbox_f2_m2[[1, 3]] * im_h

                        # xA1 = max(xmin11, xmin12);  yA1 = max(ymin11, ymin12)
                        # xB1 = min(xmax11, xmax12);  yB1 = min(ymax11, ymax12)
                        #
                        # xA2 = max(xmin21, xmin22);  yA2 = max(ymin21, ymin22)
                        # xB2 = min(xmax21, xmax22);  yB2 = min(ymax21, ymax22)

                        # if (xB1 > xA1 and yB1 > yA1) or  (xB2 > xA2 and yB2 > yA2): # there is intersection
                        tmp1 = frame1u[int(min(ymin11, ymin12)):int(max(ymax11, ymax12)), int(min(xmin11, xmin12)):int(max(xmax11, xmax12))]
                        tmp2 = frame2u[int(min(ymin21, ymin22)):int(max(ymax21, ymax22)), int(min(xmin21, xmin22)):int(max(xmax21, xmax22))]

                        if np.prod(tmp1.shape) > np.prod(tmp2.shape):
                            tmp2 = cv2.resize(tmp2, (tmp1.shape[1], tmp1.shape[0]))
                        else:
                            tmp1 = cv2.resize(tmp1, (tmp2.shape[1], tmp2.shape[0]))

                        track['data'][0, f, ind1] = (np.sum((tmp2 - tmp1) ** 2)) / float((np.sum((tmp1) ** 2)))
                        track['data'][1, f, ind1] = (np.sum((tmp2 - tmp1) ** 2)) / float((np.sum((tmp1) ** 2)))

                if 'pixel_change_front' in features:
                    track['data'][0, f, indf] = (np.sum((frame2f - frame1f) ** 2)) / float((np.sum((frame1f) ** 2)))
                    track['data'][1, f, indf] = track['data'][0, f, indf]

                if 'nose_pc' in features:
                    for s in range(2):
                        coords = track['data'][s, f - 1, :14]
                        coords2 = track['data'][s, f, :14]

                        xm1 = coords[::2]
                        ym1 = coords[1::2]

                        xm2 = coords2[::2]
                        ym2 = coords2[1::2]
                        ps = 10
                        for p in range(num_points):
                            miny1 = int(ym1[p] - ps)
                            minx1 = int(xm1[p] - ps)
                            maxy1 = int(ym1[p] + ps)
                            maxx1 = int(xm1[p] + ps)
                            while miny1 < 0: miny1 += 1
                            # while maxy1 < 0: maxy1 += 1
                            while minx1 < 0: minx1 += 1
                            # while maxx1 < 0: maxx1 += 1
                            while maxy1 > im_h: maxy1 -= 1
                            # while miny1 > im_h -1 : miny1 -= 1
                            while maxx1 > im_w: maxx1 -= 1
                            # while minx1 > im_w -1: minx1 -= 1

                            miny2 = int(ym2[p] - ps)
                            minx2 = int(xm2[p] - ps)
                            maxy2 = int(ym2[p] + ps)
                            maxx2 = int(xm2[p] + ps)
                            while miny2 < 0: miny2 += 1
                            # while maxy2 < 0: maxy2 += 1
                            while minx2 < 0: minx2 += 1
                            # while maxx2 < 0: maxx2 += 1
                            while maxy2 > im_h: maxy2 -= 1
                            # while miny2 > im_h-1: miny2 -= 1
                            while maxx2 > im_w: maxx2 -= 1
                            # while minx2 > im_w-1: minx2 -= 1

                            f1x = range(minx1, maxx1) if minx1 < maxx1 else range(maxx1, minx1)
                            f1y = range(miny1, maxy1) if miny1 < maxy1 else range(maxy1, miny1)
                            f2x = range(minx2, maxx2) if minx2 < maxx2 else range(maxx2, minx2)
                            f2y = range(miny2, maxy2) if miny2 < maxy2 else range(maxy2, miny2)
                            x1, y1 = np.ix_(f1x, f1y)
                            x2, y2 = np.ix_(f2x, f2y)
                            patch1 = frame1[y1, x1]
                            patch2 = frame2[y2, x2]
                            max_h = max(patch1.shape[0], patch2.shape[0])
                            max_w = max(patch1.shape[1], patch2.shape[1])
                            patch2 = cv2.resize(patch2, (max_w, max_h))
                            patch1 = cv2.resize(patch1, (max_w, max_h))
                            track['data'][s, f, indn + p] = (np.sum((patch2 - patch1) ** 2)) / float((np.sum((patch1) ** 2)))
                            # else:
                            #     track['data'][s, f, ind + p]=0

            if 'pixel_change' in features:
                track['data_smooth'][0, :, ind] = track['data'][0, :, ind]
                track['data_smooth'][0, 1:-1, ind] = sig.convolve(track['data_smooth'][0, :, ind], smooth_kernel, 'valid')
                track['data_smooth'][1, :, ind] = track['data_smooth'][0, :, ind]

            if 'pixel_change_front' in features:
                track['data_smooth'][0, :, indf] = track['data'][0, :, indf]
                track['data_smooth'][0, 1:-1, indf] = sig.convolve(track['data_smooth'][0, :, indf], smooth_kernel, 'valid')
                track['data_smooth'][1, :, indf] = track['data_smooth'][0, :, indf]

            if 'pixel_change_ubbox_mice' in features:
                track['data_smooth'][0, :, ind1] = track['data'][0, :, ind1]
                track['data_smooth'][0, 1:-1, ind1] = sig.convolve(track['data_smooth'][0, :, ind1], smooth_kernel, 'valid')
                track['data_smooth'][1, :, ind1] = track['data'][1, :, ind1]
                track['data_smooth'][1, 1:-1, ind1] = sig.convolve(track['data_smooth'][1, :, ind1], smooth_kernel, 'valid')

            if 'nose_pc' in features:
                for s in range(2):
                    for p in range(num_points):
                        track['data_smooth'][s, :, indn + p] = track['data'][s, :, indn + p]
                        track['data_smooth'][s, 1:-1, indn + p] = sig.convolve(track['data_smooth'][s, :, indn + p], smooth_kernel, 'valid')

            bar.finish()

        if 'resh_twd_itrhb' in features:
            ind = features.index('resh_twd_itrhb')

            def get_ell_distance(pt, theta):
                # (x,y) coordinate of the nearest point from the intruder's face to an
                # ellipse (intruder's head or body), traveling along a line at angle theta
                # theta = -pi/2 ->  3 o'clock (dead right)
                # theta = 0     -> 12 o'clock (straight ahead)
                # theta = pi/2  ->  9 o'clock (dead left)

                # (xc,yc) -> centroid of ellipse
                # M -> major axis of ellipse (radius)
                # m -> minor axis of ellipse (radius)
                # s -> orientation of ellipse
                # b -> y-intercept of line
                # k -> slope of line
                theta += np.pi / 2.

                b = 0
                k = np.tan(theta)
                xc = np.mean(pt[0, :])
                yc = np.mean(pt[1, :])
                m = distance(pt[:, 1], pt[:, 2]) / 2.
                M = distance(pt[:, 0], pt[:, 3]) / 2.
                s = pt[:, 0] - np.array([xc, yc])
                x, y = intercept_ell(xc, yc, s[1] / s[0], m, M, k, b)

                if (theta > np.pi / 2. and x >= 0) or (theta <= np.pi / 2. and x <= 0):
                    x = np.Inf
                    y = np.Inf

                if not np.isreal(x) or not np.isreal(y):
                    x = np.Inf
                    y = np.Inf

                return np.array([x, y])

            def intercept_ell(xc, yc, s, m, M, k, b):
                A = (k - s)
                B = (b - yc + s * xc)
                C = (1 + s * k)
                D = (s * b - s * yc - xc)
                w1 = m ** 2 * (1 + s ** 2)
                w2 = M ** 2 * (1 + s ** 2)

                aa = (A ** 2 / w1 + C ** 2 / w2)
                bb = (2 * A * B / w1 + 2 * C * D / w2)
                cc = (B ** 2 / w1 + D ** 2 / w2 - 1)

                x1 = (-bb + cmh.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)
                x2 = (-bb - cmh.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)

                y1 = k * x1 + b
                y2 = k * x2 + b

                if (cmh.sqrt(x1 ** 2 + y1 ** 2).real < cmh.sqrt(x2 ** 2 + y2 ** 2).real):
                    x = x1
                    y = y1
                else:
                    x = x2
                    y = y2

                return x, y

            def smooths(vin, wsize, sigma):
                wsize = np.array([1, wsize])
                overshoots = np.floor(wsize[1] / 2).astype(int)
                if overshoots * 2 == wsize[1]:
                    downshoot = overshoots - 1
                    upshoot = overshoots
                else:
                    downshoot = overshoots
                    upshoot = overshoots
                vout = vin
                x = np.arange(-(wsize[1] - 1) / 2, (wsize[1]) / 2)
                smoothfilt = np.exp(-(x * x) / (2 * sigma))
                smoothfilt = smoothfilt / sum(smoothfilt)
                for hdx in range(vin.shape[0]):
                    tvout = np.convolve(vin[hdx, :], smoothfilt, mode='full')
                    vout[hdx, :] = tvout[downshoot:len(tvout) - upshoot]

                return vout

            def distance(p1, p2):
                return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

            ######################
            keypoints = np.array(frames_pose['keypoints'])
            pts = np.zeros((len(keypoints), 2, 15))
            pts[:, :, :7] = keypoints[:, 0, :, :]
            pts[:, :, 7] = (pts[:, :, 1] + pts[:, :, 2]) / 2.
            pts[:, :, 8:] = keypoints[:, 1, :, :]
            pts -= pts[:, :, 7, np.newaxis]
            pts[:, :, 2] = smooths(pts[:, :, 2].transpose(), 125, 25).transpose()

            for fr in range(num_frames):
                # rotate all the points so the resident is facing straight ahead (positive
                # direction on y-axis). I define "straight ahead" as making the line between
                # the ears have zero slope- you could equivalently set it so the line from
                # center-eyes to nose has inf slope (or do 0 slope then rotate an additional 90
                # degrees, to avoid numerical treachery.)
                p = pts[fr]
                th = -mh.atan2(p[1, 2], p[0, 2])
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                p = R.dot(p)

                # compute distance to closest point on intruder mouse's head and body.
                # Works kind of like ray tracing- over a range of theta, ask what the
                # distance from intruder's head to closest point on the resident is, when
                # you travel along a line at orientation theta.
                #
                # theta = -pi/2 ->  3 o'clock (dead right)
                # theta = 0     -> 12 o'clock (straight ahead)
                # theta = pi/2  ->  9 o'clock (dead left)
                # range of angles to check  (can replace this with thR = 0 if you just want to check straight ahead)
                thR = np.arange(-np.pi * 3 / 4, np.pi / 4 + .01, .02)
                head = np.zeros((2, len(thR)))
                body = np.zeros((2, len(thR)))
                for i, t in enumerate(thR):
                    head[:, i] = get_ell_distance(p[:, 8:12], t)
                    body[:, i] = get_ell_distance(p[:, 11:], t)
                # convert coordinates to distances:
                dHead = np.array([distance(np.array([0, 0]), head[:, i]) for i in range(head.shape[1])])
                dBody = np.array([distance(np.array([0, 0]), body[:, i]) for i in range(body.shape[1])])
                dMouse = np.minimum(dHead, dBody)

                # find if there are point of interception
                pp = np.where(dMouse != np.Inf)[0]
                if pp.size: track['data'][:, fr, ind] = 1
                # part = np.zeros(len(dMouse))
                # for i in range(len(dMouse)):    part[i] = 1 if dMouse[i] == dHead[i] else 2
                # int_points = np.zeros((2, len(pp)))
                # for i in range(len(pp)):  int_points[:, i] = head[:, pp[i]] if part[pp[i]] == 1 else body[:, pp[i]]

            track['data_smooth'][:, :, ind] = track['data'][:, :, ind]

        del track['data']

        # augment features


        # del track['bbox']
        # del track['bbox_front']
        # del track['keypoints']
        # del track['keypoints_front']
        reader.close()
        readerf.close()
        return track
    except Exception as e:
        import linecache
        print("Error when extracting features (extract_features_top_pcf):")
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
        print(e)
        reader.close()
        readerf.close()
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
                    feat = classic_extract_features_top_pcf(top_video_fullpath=video_fullpath,
                                                            front_video_fullpath=front_video_fullpath,
                                                            top_pose_fullpath=top_pose_fullpath,
                                                            progress_bar_sig=progress_bar_sig,
                                                            max_frames=max_frames)

                elif feature_type == 'raw':
                    num_mice = 2
                    feat = classic_extract_features_top(top_video_fullpath=video_fullpath,
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
