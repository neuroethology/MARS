import numpy as np
import math as mh
import copy

def generate_valid_feature_list(cfg):
    num_mice = len(cfg['animal_names']) * cfg['num_obj']
    mice = ['m'+str(i) for i in range(num_mice)]
    # TODO: multi-camera support
    # TODO: replace hard-coded feature names with keypoints from the project config file.
    cameras = {'top':   ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base'],
               'front': ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base',
                         'left_front_paw', 'right_front_paw', 'left_rear_paw', 'right_rear_paw']}
    inferred_parts = ['centroid', 'centroid_head', 'centroid_body']

    lam = generate_lambdas()
    have_lambdas = [k for f in lam.keys() for k in lam[f].keys()]

    feats = {}
    for cam in cameras:
        pairmice = copy.deepcopy(mice)
        feats[cam] = {}
        for mouse in mice:
            feats[cam][mouse] = {}
            feats[cam][mouse]['absolute_orientation'] = ['phi', 'ori_head', 'ori_body']
            feats[cam][mouse]['joint_angle'] = ['angle_head_body_l', 'angle_head_body_r', 'angle_nose_neck_tail', 'angle_to_center']
            feats[cam][mouse]['joint_angle_trig'] = ['sin_angle_head_body_l', 'cos_angle_head_body_l',
                                                     'sin_angle_head_body_r', 'cos_angle_head_body_r',
                                                     'sin_angle_nose_neck_tail', 'cos_angle_nose_neck_tail']
            feats[cam][mouse]['fit_ellipse'] = ['major_axis_len', 'minor_axis_len', 'axis_ratio', 'area_ellipse']
            feats[cam][mouse]['distance_to_walls'] = ['dist_edge_x', 'dist_edge_y', 'dist_edge', 'dist_to_center']
            feats[cam][mouse]['speed'] = ['speed', 'speed_centroid', 'speed_fwd', 'max_jitter', 'mean_jitter']
            feats[cam][mouse]['acceleration'] = ['acceleration_head', 'acceleration_body', 'acceleration_centroid']

        pairmice.remove(mouse)
        for mouse2 in pairmice:
            feats[cam][mouse2 + mouse] = {}
            feats[cam][mouse2 + mouse]['social_angle'] = ['angle_between', 'facing_angle', 'angle_social']
            feats[cam][mouse2 + mouse]['social_angle_trig'] = ['sin_angle_between', 'cos_angle_between', 'sin_facing_angle', 'cos_facing_angle', 'sin_angle_social', 'cos_angle_social']
            feats[cam][mouse2 + mouse]['relative_size'] = ['area_ellipse_ratio']
            feats[cam][mouse2 + mouse]['social_distance'] = ['dist_centroid', 'dist_nose', 'dist_head', 'dist_body',
                                                   'dist_head_body', 'dist_gap', 'dist_scaled', 'overlap_bboxes']

    for cam, parts in zip(cameras.keys(), [cameras[i] for i in cameras.keys()]):
        for mouse in mice:
            feats[cam][mouse]['raw_coordinates'] = [(p + c) for p in parts for c in ['_x', '_y']]
            [feats[cam][mouse]['raw_coordinates'].append(p + c) for p in inferred_parts for c in ['_x', '_y']]
            feats[cam][mouse]['intramouse_distance'] = [('dist_' + p + '_' + q) for i, p in enumerate(parts) for q in parts[i+1:]]
        for mouse2 in pairmice:
            feats[cam][mouse2 + mouse]['intermouse_distance'] = [('dist_m0' + p + '_m1' + q) for p in parts for q in parts]

    # make sure we have a lambda for each named feature
    for cam in cameras:
        for mouse in feats[cam].keys():
            for grp in feats[cam][mouse].keys():
                feats[cam][mouse][grp] = [f for f in feats[cam][mouse][grp] if f in have_lambdas]

    return feats

def generate_lambdas():
    # define the lambdas for all the features, grouped by their required inputs.
    # units for all lambdas are in pixels and frames. These should be converted to mouselengths and seconds
    # by the extract_features function.

    eps = np.spacing(1)
    parts_list = ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base', 'left_front_paw',
                   'right_front_paw', 'left_rear_paw', 'right_rear_paw']

    # lambdas are grouped by what kind of input they take. not very intuitive naming, this is supposed
    # to be behind the scenes. if you want to play with which features you compute, modify the groups
    # in generate_feature_list.
    lam = {'ell_ang': {}, 'ell': {}, 'ell_area': {}, 'xy_ang': {}, 'xy': {}, 'xybd': {}, 'xybd_ang':{}, 'dt': {}, '2mdt': {},
           'd2t': {}, 'xyxy_ang': {}, 'xyxy': {}, 'bb': {}, 'video': {}, 'bb_video': {}, 'xy_ang_trig': {},
           'xyxy_ang_trig': {}}

    # features based on a fit ellipse ###################################################
    lam['ell_ang']['phi'] = lambda ell: ell['phi']
    lam['ell']['major_axis_len'] = lambda ell: ell['ra'] if ell['ra'] > 0. else eps
    lam['ell']['minor_axis_len'] = lambda ell: ell['rb'] if ell['rb'] > 0. else eps
    lam['ell_ang']['axis_ratio'] = lambda ell: ell['ra'] / ell['rb'] if ell['rb'] > 0. else eps
    lam['ell_area']['area_ellipse'] = lambda ell: mh.pi * ell['ra'] * ell['rb'] if ell['ra'] * ell['rb'] > 0. else eps

    # features based on the location of one mouse #######################################
    lam['xy_ang']['ori_head'] = lambda x, y: get_angle(x[3], y[3], x[0], y[0])
    lam['xy_ang']['ori_body'] = lambda x, y: get_angle(x[6], y[6], x[3], y[3])
    lam['xy_ang']['angle_head_body_l'] = lambda x, y: interior_angle([x[2], y[2]], [x[3], y[3]], [x[5], y[5]])
    lam['xy_ang']['angle_head_body_r'] = lambda x, y: interior_angle([x[1], y[1]], [x[3], y[3]], [x[4], y[4]])
    lam['xy_ang']['angle_nose_neck_tail'] = lambda x, y: interior_angle_orth([x[0], y[0]], [x[3], y[3]], [x[6], y[6]])

    # add sines and cosines of angle features
    # for k in list(lam['xy_ang'].keys()):
    #     lam['xy_ang_trig']['sin_' + k] = lambda x, y: np.sin(lam['xy_ang'][k](x, y))
    #     lam['xy_ang_trig']['cos_' + k] = lambda x, y: np.cos(lam['xy_ang'][k](x, y))
    lam['xy_ang']['sin_ori_head'] = lambda x, y: np.sin(lam['xy_ang']['ori_head'](x, y))
    lam['xy_ang']['cos_ori_head'] = lambda x, y: np.cos(lam['xy_ang']['ori_head'](x, y))
    lam['xy_ang']['sin_ori_body'] = lambda x, y: np.sin(lam['xy_ang']['ori_body'](x, y))
    lam['xy_ang']['cos_ori_body'] = lambda x, y: np.cos(lam['xy_ang']['ori_body'](x, y))
    lam['xy_ang']['sin_angle_head_body_l'] = lambda x, y: np.sin(lam['xy_ang']['angle_head_body_l'](x, y))
    lam['xy_ang']['cos_angle_head_body_l'] = lambda x, y: np.cos(lam['xy_ang']['angle_head_body_l'](x, y))
    lam['xy_ang']['sin_angle_head_body_r'] = lambda x, y: np.sin(lam['xy_ang']['angle_head_body_r'](x, y))
    lam['xy_ang']['cos_angle_head_body_r'] = lambda x, y: np.cos(lam['xy_ang']['angle_head_body_r'](x, y))
    lam['xy_ang']['sin_angle_nose_neck_tail'] = lambda x, y: np.sin(lam['xy_ang']['angle_nose_neck_tail'](x, y))
    lam['xy_ang']['cos_angle_nose_neck_tail'] = lambda x, y: np.cos(lam['xy_ang']['angle_nose_neck_tail'](x, y))


    lam['xy']['centroid_x'] = lambda x, y: np.mean(x)
    lam['xy']['centroid_y'] = lambda x, y: np.mean(y)
    lam['xy']['centroid_head_x'] = lambda x, y: np.mean(x[:3])
    lam['xy']['centroid_head_y'] = lambda x, y: np.mean(y[:3])
    lam['xy']['centroid_body_x'] = lambda x, y: np.mean(x[4:])
    lam['xy']['centroid_body_y'] = lambda x, y: np.mean(y[4:])
    for i, p1 in enumerate(parts_list):
        for j, p2 in enumerate(parts_list[i+1:]):
            lam['xy']['dist_' + p1 + '_' + p2] = lambda x, y, ind1=i, ind2=j: \
                                                        np.linalg.norm([x[ind1] - x[ind2], y[ind1] - y[ind2]])
    for i, part in enumerate(parts_list):
        lam['xy'][part + '_x'] = lambda x, y, ind=i: x[ind]
        lam['xy'][part + '_y'] = lambda x, y, ind=i: y[ind]

    # features based on position or angle w.r.t. arena ###########################################
    lam['xybd']['_center'] = lambda x, y, xlims, ylims: np.sin(interior_angle_orth([x[0], y[0]], [x[3], y[3]],
                                                                                    [(xlims[1] - xlims[0]) / 2 + xlims[0],
                                                                                     (ylims[1] - ylims[0]) / 2 + ylims[0]]))
    lam['xybd']['dist_to_center'] = lambda x, y, xlims, ylims: np.linalg.norm([x[0] - ((xlims[1] - xlims[0]) / 2 + xlims[0]),
                                                                               y[0] - ((ylims[1] - ylims[0]) / 2 + ylims[0])])
    lam['xybd']['dist_edge_x'] = lambda x, y, xlims, ylims:\
        np.amin(np.stack((np.maximum(0, lam['xy']['centroid_x'](x, y) - xlims[0]),
                          np.maximum(0, xlims[1] - lam['xy']['centroid_x'](x, y))), axis=-1), axis=0)
    lam['xybd']['dist_edge_y'] = lambda x, y, xlims, ylims: \
        np.amin(np.stack((np.maximum(0, lam['xy']['centroid_y'](x, y) - ylims[0]),
                          np.maximum(0, ylims[1] - lam['xy']['centroid_y'](x, y))), axis=-1), axis=0)
    lam['xybd']['dist_edge'] = lambda x, y, xlims, ylims:\
        np.amin(np.stack((lam['xybd']['dist_edge_x'](x, y, xlims, ylims),
                          lam['xybd']['dist_edge_y'](x, y, xlims, ylims)), axis=-1), axis=0)

    # velocity features #################################################################
    # question: should we instead estimate velocities with a kalman filter, to reduce noise?
    lam['dt']['speed'] = lambda xt1, yt1, xt2, yt2: speed_head_hips(lam, xt1, yt1, xt2, yt2)
    lam['dt']['speed_centroid'] = lambda xt1, yt1, xt2, yt2: speed_centroid(lam, xt1, yt1, xt2, yt2)
    lam['dt']['speed_fwd'] = lambda xt1, yt1, xt2, yt2: speed_fwd(lam, xt1, yt1, xt2, yt2)
    lam['dt']['mean_jitter'] = lambda xt1, yt1, xt2, yt2: np.mean(np.linalg.norm([xt2 - xt1, yt2 - yt1], axis=0))
    lam['dt']['max_jitter'] = lambda xt1, yt1, xt2, yt2: np.max(np.linalg.norm([xt2 - xt1, yt2 - yt1], axis=0))
    # going to omit the windowed [('speed_centroid_' + w) for w in ['w2', 'w5', 'w10']],
    # as these are too sensitive to changes in imaging framerate

    # social velocity features ##########################################################
    lam['2mdt']['radial_vel'] = lambda xt2, yt2, xt1, yt1, x2, y2: radial_vel(lam, xt2, yt2, xt1, yt1, x2, y2)
    lam['2mdt']['tangential_vel'] = lambda xt2, yt2, xt1, yt1, x2, y2: tangential_vel(lam, xt2, yt2, xt1, yt1, x2, y2)

    # acceleration features #############################################################
    lam['d2t']['acceleration_head'] = lambda x2, y2, x1, y1, x0, y0: acceleration_head(lam, x2, y2, x1, y1, x0, y0)
    lam['d2t']['acceleration_body'] = lambda x2, y2, x1, y1, x0, y0: acceleration_body(lam, x2, y2, x1, y1, x0, y0)
    lam['d2t']['acceleration_centroid'] = lambda x2, y2, x1, y1, x0, y0: acceleration_ctr(lam, x2, y2, x1, y1, x0, y0)

    # features based on the locations of both mice ######################################
    lam['xyxy_ang']['facing_angle'] = lambda x1, y1, x2, y2: facing_angle(lam, x1, y1, x2, y2)
    lam['xyxy_ang']['angle_between'] = lambda x1, y1, x2, y2: angle_between(lam, x1, y1, x2, y2)
    lam['xyxy_ang']['angle_social'] = lambda x1, y1, x2, y2: soc_angle(lam, x1, y1, x2, y2)

    # for k in list(lam['xyxy_ang'].keys()):
    #     lam['xyxy_ang_trig']['sin_' + k] = lambda x1, y1, x2, y2: np.sin(lam['xyxy_ang'][k](x1, y1, x2, y2))
    #     lam['xyxy_ang_trig']['cos_' + k] = lambda x1, y1, x2, y2: np.cos(lam['xyxy_ang'][k](x1, y1, x2, y2))
    lam['xyxy_ang_trig']['sin_facing_angle'] = lambda x1, y1, x2, y2: np.sin(lam['xyxy_ang']['facing_angle'](x1, y1, x2, y2))
    lam['xyxy_ang_trig']['cos_facing_angle'] = lambda x1, y1, x2, y2: np.cos(lam['xyxy_ang']['facing_angle'](x1, y1, x2, y2))
    lam['xyxy_ang_trig']['sin_angle_between'] = lambda x1, y1, x2, y2: np.sin(lam['xyxy_ang']['angle_between'](x1, y1, x2, y2))
    lam['xyxy_ang_trig']['cos_angle_between'] = lambda x1, y1, x2, y2: np.cos(lam['xyxy_ang']['angle_between'](x1, y1, x2, y2))
    lam['xyxy_ang_trig']['sin_angle_social'] = lambda x1, y1, x2, y2: np.sin(lam['xyxy_ang']['angle_social'](x1, y1, x2, y2))
    lam['xyxy_ang_trig']['cos_angle_social'] = lambda x1, y1, x2, y2: np.cos(lam['xyxy_ang']['angle_social'](x1, y1, x2, y2))

    lam['xyxy']['dist_nose'] = lambda x1, y1, x2, y2: dist_nose(lam, x1, y1, x2, y2)
    lam['xyxy']['dist_body'] = lambda x1, y1, x2, y2: dist_body(lam, x1, y1, x2, y2)
    lam['xyxy']['dist_head'] = lambda x1, y1, x2, y2: dist_head(lam, x1, y1, x2, y2)
    lam['xyxy']['dist_centroid'] = lambda x1, y1, x2, y2: dist_centroid(lam, x1, y1, x2, y2)
    lam['xyxy']['dist_head_body'] = lambda x1, y1, x2, y2: dist_head_body(lam, x1, y1, x2, y2)
    lam['xyxy']['dist_gap'] = lambda x1, y1, x2, y2: dist_gap(lam, x1, y1, x2, y2)
    lam['xyxy_ang']['area_ellipse_ratio'] = lambda x1, y1, x2, y2: \
        lam['ell_area']['area_ellipse'](fit_ellipse(x1, y1))/lam['ell_area']['area_ellipse'](fit_ellipse(x2, y2))

    for i, p1 in enumerate(parts_list):
        for j, p2 in enumerate(parts_list):
            lam['xyxy']['dist_m1' + p1 + '_m2' + p2] = \
                lambda x1, y1, x2, y2, ind1=i, ind2=j: np.linalg.norm([x1[ind1] - x2[ind2], y1[ind1] - y2[ind2]])

    # features based on the bounding boxes ##############################################
    lam['bb']['overlap_bboxes'] = lambda box1, box2: bb_intersection_over_union(box1, box2)

    # features based on video frames ####################################################
    lam['video']['pix_change_local'] = lambda img1, img2, x1, y1, x2, y2, l: pixel_change_local(lam, img1, img2, x1, y1, x2, y2, l)

    # feature based on bounding boxes and video frames, this one is dumb ################
    lam['bb_video']['pix_change_bbox'] = lambda img1, img2, bb1, bb2, bb10, bb20: pixel_change_ubbox(lam, bb1, bb2, bb10, bb20, img1, img2)

    return lam


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
    # def unit_vector(v):
    #     return v/np.linalg.norm(v)
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    ang = mh.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return ang


def interior_angle_orth(p0, p1, p2):
    # def unit_vector(v):
    #     return v/np.linalg.norm(v)
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    ang = -mh.atan2(np.dot(v0, v1), np.linalg.det([v0, v1])) # flip X/Y. use this if your angles are hovering around pi to reduce flippage
    ang = ang if ang>0 else 2*mh.pi+ang
    ang = -(ang - mh.pi/2) + mh.pi
    ang = ang if ang<=2*mh.pi else ang-(2*mh.pi)
    return ang


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
