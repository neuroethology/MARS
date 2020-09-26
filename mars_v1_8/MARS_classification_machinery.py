import numpy as np
import os
from hmmlearn import hmm
from scipy import signal
from sklearn.preprocessing import StandardScaler
import pdb
import datetime as dtime
import stat
import multiprocessing
from time import time as tt
from sklearn.preprocessing import binarize
import xlwt
import joblib
import MARS_output_format as mof

def get_rel_path(path, start=''):
    return './' + os.path.relpath(path,start)


flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))

def clean_data(data):
    """Eliminate the NaN and Inf values by taking the last value that was neither."""
    idx = np.where(np.isnan(data) | np.isinf(data))
    if idx[0].size>0:
        for j in range(len(idx[0])):
            if idx[0][j] == 0:
                data[idx[0][j], idx[1][j],idx[2][j]] = 0.
            else:
                data[idx[0][j], idx[1][j],idx[2][j]] = data[idx[0][j] - 1, idx[1][j],idx[2][j]]
    return data

def normalize_pixel_data(data,view):
    if view=='top':fd = [range(40, 49)]
    elif view=='front': fd=[range(47,67)]
    elif view =='top_pcf':fd=[range(40,57)]
    fd = list(flatten(fd))
    md = np.nanmedian(data[:, :, fd], 1, keepdims=True)
    data[:, :, fd] /= md
    return data

def do_fbs(y_pred_class, kn, blur, blur_steps, shift):
    """Does forward-backward smoothing."""
    len_y = len(y_pred_class)

    # fbs with classes
    z = np.zeros((3, len_y))  # Make a matrix to hold the shifted predictions --one row for each shift.

    # Create mirrored start and end indices for extending the length of our prediction vector.
    mirrored_start = range(shift, -1, -1)  # Creates indices that go (shift, shift-1, ..., 0)
    mirrored_end = range(len_y - 1, len_y - 1 - shift, -1)  # Creates indices that go (-1, -2, ..., -shift)

    # Now we extend the predictions to have a mirrored portion on the front and back.
    extended_predictions = np.r_[
        y_pred_class[mirrored_start],
        y_pred_class,
        y_pred_class[mirrored_end]
    ]

    # Do our blurring.
    for s in range(blur_steps):
        extended_predictions = signal.convolve(np.r_[extended_predictions[0],
                                                     extended_predictions,
                                                     extended_predictions[-1]],
                                               kn / kn.sum(),  # The kernel we are convolving.
                                               'valid')  # Only use valid conformations of the filter.
        # Note: this will leave us with 2 fewer items in our signal each iteration, so we append on both sides.

    z[0, :] = extended_predictions[2 * shift + 1:]
    z[1, :] = extended_predictions[:-2 * shift - 1]
    z[2, :] = extended_predictions[shift + 1:-shift]

    z_mean = np.mean(z, axis=0)  # Average the blurred and shifted signals together.

    y_pred_fbs = binarize(z_mean.reshape((-1, 1)), .5).astype(int).reshape((1, -1))[0]  # Anything that has a signal strength over 0.5, is taken to be positive.

    return y_pred_fbs


def load_features_from_filename(top_feat_name='', front_feat_name=''):
    try:
        if top_feat_name and front_feat_name:
            # Load the features using the combined function.
            if 'wnd' in top_feat_name and 'wnd' in front_feat_name:
                features = load_features_both_wnd(top_feat_name, front_feat_name)
            else:
                features = load_features_both(top_feat_name, front_feat_name)
        elif top_feat_name:
            # Load top features.
            if 'wnd' in top_feat_name:
                features = load_features_top_wnd(top_feat_name)
            else:
                if 'pcf' in top_feat_name:
                    features = load_features_top_pcf(top_feat_name)

                else:
                    features = load_features_top(top_feat_name)
        else:
            err_msg = "Trying to extract features, but no valid feature-names provided."
            print(err_msg)
            raise ValueError(err_msg)
    except Exception as e:
        print(e)
        raise(e)

    return features

def load_features_front(front_feat_name):
    flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
    vid = np.load(front_feat_name)
    d = vid['data_smooth']
    n_feat=d.shape[2]
    features = clean_data(d)
    features = normalize_pixel_data(features,'front')
    features = clean_data(features)
    featToKeep = list(flatten([range(47), range(56,74), 75,77,78,79, range(201, n_feat)]))
    features = np.hstack((features[0, :, :], features[1, :, featToKeep].transpose()))

    print('front features loaded')
    return features

def load_features_front_wnd(front_feat_name):
    features = np.load(front_feat_name)['data']
    return features

def load_features_top(top_feat_name):

    flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
    vid = np.load(top_feat_name)
    d = vid['data_smooth']
    n_feat = d.shape[2]
    features = clean_data(d)
    features = normalize_pixel_data(features, 'top')
    features = clean_data(features)
    featToKeep = list(flatten([range(39), range(42, 58), 59, 61, 62, 63, range(113, n_feat)]))
    features = np.hstack((features[0, :, :], features[1, :, featToKeep].transpose()))

    print('top features loaded')
    return features

def load_features_top_wnd(top_feat_name):
    features = np.load(top_feat_name)['data']
    return features

def load_features_top_pcf(top_feat_name):

    flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
    vid = np.load(top_feat_name)
    d = vid['data_smooth']
    n_feat = d.shape[2]
    features = clean_data(d)
    features = normalize_pixel_data(features, 'top')
    features = clean_data(features)
    featToKeep = list(flatten([range(39), range(50, 66), 67, 69, 70, 71, range(121, n_feat)]))
    features = np.hstack((features[0, :, :], features[1, :, featToKeep].transpose()))

    print('top features loaded')
    return features



def load_features_both(top_feat_name, front_feat_name):
    features_top  = load_features_top(top_feat_name)
    features_front = load_features_front(front_feat_name)
    features = np.concatenate((features_top, features_front), axis=1)

    print('top and front features_loaded')
    return features

def load_features_both_wnd(top_feat_name, front_feat_name):
    features_top  = load_features_top_wnd(top_feat_name)['data']
    features_front = load_features_front_wnd(front_feat_name)['data']
    features = np.concatenate((features_top, features_front), axis=1)

    print('top and front features_loaded')
    return features


def predict_labels(features, classifier_path, behaviors=[]):
    print("predicting probabilities")
    all_predicted_probabilities, behaviors_used = predict_probabilities(features=features,
                                                                        classifier_path=classifier_path,
                                                                        behaviors=behaviors)
    print("assigning labels")
    labels,labels_iteraction = assign_labels(all_predicted_probabilities=all_predicted_probabilities,
                           behaviors_used=behaviors_used)
    return labels,labels_iteraction

def predict_probabilities(features, classifier_path, behaviors=[], VERBOSE = True):
        ''' This predicts behavior labels on a video, given a classifier and features.'''
        if not behaviors:
            behaviors = ['closeinvestigation','mount', 'attack','interaction']


        scaler = joblib.load(classifier_path + '/scaler')
        # Scale the data appropriately.
        print("transforming features")
        X_test = scaler.transform(features)

        print("assembling models")
        models = [os.path.join(classifier_path, filename) for filename in os.listdir(classifier_path)]
        behaviors_used = []

        preds_fbs_hmm = []
        proba_fbs_hmm = []

        print("predict_probabilities: just before for loop")
        for b, behavior in enumerate(behaviors):
            # For each behavior, load the model, load in the data (if needed), and predict on it.
            print('############################## %s #########################' % behavior)

            # Get all the models that model the given behavior.
            models_with_this_behavior = filter(lambda x: x.find('classifier_' + behavior) > -1, models)
            # pdb.set_trace()
            # If there are models for this behavior, load the most recently trained one.
            if models_with_this_behavior:
                # create a dict that contains list of files and their modification timestamps
                name_n_timestamp = dict([(x, os.stat(x).st_mtime) for x in models_with_this_behavior])
                # return the file with the latest timestamp
                name_classifier = max(name_n_timestamp, key=lambda k: name_n_timestamp.get(k))

                classifier = joblib.load(name_classifier)

                bag_clf = classifier['bag_clf']  if 'bag_clf' in classifier.keys() else classifier['clf']
                hmm_fbs = classifier['hmm_fbs']
                kn = classifier['k']
                blur_steps = classifier['blur_steps']
                shift = classifier['shift']

                # Keep track of which behaviors get used.
                behaviors_used += [behavior]

            else:
                print('Classifier not found, you need to train a classifier for this behavior before using it')
                print('Classification will continue without classifying this behavior')
                continue


            if VERBOSE:
                print("Predicting...")
                tstart = tt()

            # Do the actual prediction.
            predicted_probabilities = bag_clf.predict_proba(X_test)
            predicted_class = np.argmax(predicted_probabilities, axis=1)

            # if VERBOSE:
            #     secs_elapsed = (tt() - tstart)
            #     print("Classifier prediction took %.2f secs" % secs_elapsed)
            #
            #     print("Doing Forward-Backward Smoothing...")
            #     tstart = tt()

            # Do our forward-backward smoothing

            y_pred_fbs = do_fbs(y_pred_class=predicted_class, kn=kn, blur=4, blur_steps=blur_steps, shift=shift)
            # TODO: Blur is unused argument --just get rid of it?

            # Do the hmm prediction.
            y_proba_fbs_hmm = hmm_fbs.predict_proba(y_pred_fbs.reshape((-1, 1)))
            y_pred_fbs_hmm = np.argmax(y_proba_fbs_hmm, axis=1)

            # Add our predictions to the list.
            preds_fbs_hmm.append(y_pred_fbs_hmm)
            proba_fbs_hmm.append(y_proba_fbs_hmm)

            if VERBOSE:
                secs_elapsed = (tt() - tstart)
                print("Classifier prediction took %.2f secs" % secs_elapsed)

        # Change the list of [1x(numFrames)]-predictions to an np.array by stacking them vertically.
        preds_fbs_hmm = np.vstack(preds_fbs_hmm)

        # Flip it over so that it's stored as a [(numFrames)x(numBehaviors)] array
        all_predictions = preds_fbs_hmm.T

        # Change [(behavior)x(frames)x(positive/neg)] => [(frames) x (behaviors) x (pos/neg)]
        all_predicted_probabilities = np.array(proba_fbs_hmm).transpose(1, 0, 2)
        # pdb.set_trace()
        return all_predicted_probabilities, behaviors_used

def assign_labels(all_predicted_probabilities, behaviors_used):
    ''' Assigns labels based on the provided probabilities.'''
    labels = []
    labels_interaction=[]
    num_frames = all_predicted_probabilities.shape[0]
    # Looping over frames, determine which annotation label to take.
    for i in xrange(num_frames):
        # Get the [3x2] matrix of current prediction probabilities.
        current_prediction_probabilities = all_predicted_probabilities[i,:-1]
        current_prediction_probabilities_interaction = all_predicted_probabilities[i,-1]
        # Get the positive/negative labels for each behavior, by taking the argmax along the pos/neg axis.
        onehot_class_predictions = np.argmax(current_prediction_probabilities, axis=1)
        onehot_class_predictions_interaction = np.argmax(current_prediction_probabilities_interaction)
        if onehot_class_predictions_interaction == 0:
            labels_interaction.append('other')
        else:labels_interaction.append(behaviors_used[-1])

        # Get the actual probabilities of those predictions.
        predicted_class_probabilities = np.max(current_prediction_probabilities, axis=1)

        # If every behavioral predictor agrees that the current_
        if np.all(onehot_class_predictions == 0):
            # The index here is one past any positive behavior --this is how we code for "other".
            beh_frame = 0
            # How do we get the probability of it being "other?" Since everyone's predicting it, we just take the mean.
            proba_frame = np.mean(predicted_class_probabilities)
            labels += ['other']
        else:
            # If we have positive predictions, we find the probabilities of the positive labels and take the argmax.
            pos = np.where(onehot_class_predictions)[0]
            # print(pos)
            # pdb.set_trace()
            max_prob = np.argmax(predicted_class_probabilities[pos])

            # This argmax is, by construction, the id for this behavior.
            beh_frame = pos[max_prob]
            # We also want to save that probability,maybe.
            proba_frame = predicted_class_probabilities[beh_frame]
            labels += [behaviors_used[beh_frame]]
            beh_frame += 1



    return labels,labels_interaction


def is_gt_annotation(filename):
    cond1 = filename.endswith('anno.txt')
    cond2 = filename.endswith('TK.txt')
    cond3 = filename.endswith('Teresa.txt')
    return cond1|cond2|cond3

def get_annotation_keys():
    annotation_keys = ['other o',
                        'closeinvestigation i', 'sniff_genital g', 'sniff_body b', 'sniff_face f',
                        'mount m', 'mount_attempt n',
                        'intromission r',
                        'attack a', 'attack_attempt k',
                        'approach p',
                        'intruder_enter e', 'intruder_out u',
                        'cable_fix f', 'alone l',
                        'interaction c']
    return annotation_keys

def dump_labels_CBA(labels,labels_interaction, classification_txtname):
    annotation_keys = get_annotation_keys()
    lab2k = '\n'.join(annotation_keys)
    lab2k += '\n'
    # lab2k = {0: 'o', 1: 'i', 2: 'm', 3: 'a', 6: 'g', 4: 'h', 5: 'b'}

    # Open the file you want to write to.
    fp = open(classification_txtname, 'wb')

    # Write the header.
    fp.write('Caltech Behavior Annotator - Annotation File\n\n')
    fp.write('Configuration file:\n')
    fp.write(lab2k)

    # Start writing channel 1.
    fp.write('\nS1: start    end     type\n')
    fp.write('-----------------------------\n')

    #####################################################

    curr_frame_num = 0
    beginning_of_current_bout = 0
    end_of_current_bout = 0
    bouts = []
    while curr_frame_num < len(labels) - 1:
        # Get the current label and the next.
        current_label = labels[curr_frame_num]
        next_label = labels[curr_frame_num + 1]

        # If the current label is different from the next, mark this as the end of the bout.
        if current_label != next_label:
            end_of_current_bout = curr_frame_num
            # Remember: when writing to the file, the CBA format uses 1-indexed numbers and we use 0-indexed. So add 1.
            fp.write('%8i   %6i     %s\n' % (beginning_of_current_bout + 1,
                                             end_of_current_bout + 1,
                                             current_label))

            # Update where the next bout begins.
            beginning_of_current_bout = curr_frame_num + 1

        # Go to the next frame.
        curr_frame_num += 1

    # Write the last bout.
    fp.write('%8i   %6i     %s\n' % (beginning_of_current_bout + 1, len(labels), next_label))

    # Just write the second channel as nothing.
    fp.write('\nS2: start    end     type\n')
    fp.write('-----------------------------\n')

    curr_frame_num = 0
    beginning_of_current_bout = 0
    end_of_current_bout = 0
    bouts = []
    while curr_frame_num < len(labels_interaction) - 1:
        # Get the current label and the next.
        current_label = labels_interaction[curr_frame_num]
        next_label = labels_interaction[curr_frame_num + 1]

        # If the current label is different from the next, mark this as the end of the bout.
        if current_label != next_label:
            end_of_current_bout = curr_frame_num
            # Remember: when writing to the file, the CBA format uses 1-indexed numbers and we use 0-indexed. So add 1.
            fp.write('%8i   %6i     %s\n' % (beginning_of_current_bout + 1,
                                             end_of_current_bout + 1,
                                             current_label))

            # Update where the next bout begins.
            beginning_of_current_bout = curr_frame_num + 1

        # Go to the next frame.
        curr_frame_num += 1

    # Write the last bout.
    fp.write('%8i   %6i     %s\n' % (beginning_of_current_bout + 1, len(labels_interaction), next_label))

    fp.close()
    return

def dump_bento(video_fullpath, output_suffix='', pose_file = '', basepath = ''):
    if not output_suffix:
            # Default suffix is just the version number.
        output_suffix = mof.get_version_suffix()
    video_path = os.path.dirname(video_fullpath)
    video_name = os.path.basename(video_fullpath)

    # Get the output folder for this specific mouse.
    output_folder = mof.get_mouse_output_dir(dir_output_should_be_in=video_path, video_name=video_name,
                                             output_suffix=output_suffix)
    mouse_name = output_folder.split('/')[-1]

    # if not movie_file:
    #     movie_name = mouse_name + '_Top_J85.seq'
    #
    #     movie_location = output_folder
    #     movie_location = os.path.split(movie_location)[0]
    #     movie_location = os.path.split(movie_location)[0]
    #
    #     movie_file = os.path.join(movie_location, movie_name)
    # else:
        # movie_file = os.path.basename(movie_file)

    if not pose_file:
        pose_basename = mof.get_pose_no_ext(video_fullpath=video_fullpath,
                                            output_folder=output_folder,
                                            view='top',
                                            output_suffix=output_suffix)

        top_pose_fullpath = pose_basename + '.mat'
    # else:
        # pose_file = os.path.basename(pose_file)
    """ This function writes an xls with information for bento in it."""
    wb = xlwt.Workbook(encoding='utf-8')
    ws1 = wb.add_sheet('Sheet1', cell_overwrite_ok=True)
    ws1.write(0, 0, basepath ) # A1
    ws1.write(0, 1, 'Ca framerate:')  # B1
    ws1.write(0, 2, 0)  # C1
    ws1.write(0, 3, 'Annot framerate:')  # D1
    ws1.write(0, 4, 30)  # E1
    ws1.write(0, 5, 'Multiple trials/Ca file:')  # F1
    ws1.write(0, 6, 0)  # G1
    ws1.write(0, 7, 'Multiple trails/annot file')  # H1
    ws1.write(0, 8, 0)  # I1
    ws1.write(0, 9, 'Includes behavior movies:')  # J1
    ws1.write(0, 10, 1)  # K1
    ws1.write(0, 11, 'Offset (in seconds; positive values = annot starts before Ca):')  # L1
    ws1.write(0, 12, 0)  # M1

    ws1.write(1, 0, 'Mouse')  # A2
    ws1.write(1, 1, 'Sessn')  # B2
    ws1.write(1, 2, 'Trial')  # C2
    ws1.write(1, 3, 'Stim')  # D2
    ws1.write(1, 4, 'Calcium imaging file')  # E2
    ws1.write(1, 5, 'Start Ca')  # F2
    ws1.write(1, 6, 'Stop Ca')  # G2
    ws1.write(1, 7, 'FR Ca')  # H2
    ws1.write(1, 8, 'Alignments')  # I2
    ws1.write(1, 9, 'Annotation file')  # J2
    ws1.write(1, 10, 'Start Anno')  # K2
    ws1.write(1, 11, 'Stop Anno')  # L2
    ws1.write(1, 12, 'FR Anno')  # M2
    ws1.write(1, 13, 'Offset')  # N2
    ws1.write(1, 14, 'Behavior movie')  # O2
    ws1.write(1, 15, 'Tracking')  # P2

    ws1.write(2, 0, 1)  # A2
    ws1.write(2, 1, 1)  # B2
    ws1.write(2, 2, 1)  # C2
    ws1.write(2, 3, '')  # D2
    ws1.write(2, 4, '')  # E2
    ws1.write(2, 5, '')  # F2
    ws1.write(2, 6, '')  # G2
    ws1.write(2, 7, '')  # H2
    ws1.write(2, 8, '')  # I2
    ann = [os.path.join(output_folder,f)
           for f in os.listdir(output_folder) if is_gt_annotation(f)|('pred' in f)]
    ann = sorted(ann)
    ann = [get_rel_path(annot_path, basepath) for annot_path in ann ]
    ws1.write(2, 9, ';'.join(ann))  # J2
    ws1.write(2, 10, '')  # K2
    ws1.write(2, 11, '')  # L2
    ws1.write(2, 12, '')  # M2
    ws1.write(2, 13, '')  # N2
    ws1.write(2, 14, get_rel_path(video_fullpath,basepath))  # O2
    ws1.write(2, 15, get_rel_path(top_pose_fullpath,basepath))  # P2

    bento_name = 'bento_' + output_suffix +'.xls'
    wb.save(os.path.join(output_folder,bento_name))
    return 1


# def dump_bento(video_fullpath, output_suffix='', pose_file = ''):
#     if not output_suffix:
#             # Default suffix is just the version number.
#         output_suffix = mof.get_version_suffix()
#     video_path = os.path.dirname(video_fullpath)
#     video_name = os.path.basename(video_fullpath)
#
#     # Get the output folder for this specific mouse.
#     output_folder = mof.get_mouse_output_dir(dir_output_should_be_in=video_path, video_name=video_name,
#                                              output_suffix=output_suffix)
#     mouse_name = output_folder.split('/')[-1]
#
#     # if not movie_file:
#     #     movie_name = mouse_name + '_Top_J85.seq'
#     #
#     #     movie_location = output_folder
#     #     movie_location = os.path.split(movie_location)[0]
#     #     movie_location = os.path.split(movie_location)[0]
#     #
#     #     movie_file = os.path.join(movie_location, movie_name)
#     # else:
#         # movie_file = os.path.basename(movie_file)
#
#     if not pose_file:
#         pose_basename = mof.get_pose_no_ext(video_fullpath=video_fullpath,
#                                             output_folder=output_folder,
#                                             view='top',
#                                             output_suffix=output_suffix)
#
#         top_pose_fullpath = pose_basename + '.mat'
#     # else:
#         # pose_file = os.path.basename(pose_file)
#
#     """ This function writes an xls with information for bento in it."""
#     wb = xlwt.Workbook(encoding='utf-8')
#     ws1 = wb.add_sheet('Sheet1', cell_overwrite_ok=True)
#     ws1.write(0, 0, os.path.abspath('/') ) # A1
#     ws1.write(0, 1, 'Ca framerate:')  # B1
#     ws1.write(0, 2, 0)  # C1
#     ws1.write(0, 3, 'Annot framerate:')  # D1
#     ws1.write(0, 4, 30)  # E1
#     ws1.write(0, 5, 'Multiple trials/Ca file:')  # F1
#     ws1.write(0, 6, 0)  # G1
#     ws1.write(0, 7, 'Multiple trails/annot file')  # H1
#     ws1.write(0, 8, 0)  # I1
#     ws1.write(0, 9, 'Includes behavior movies:')  # J1
#     ws1.write(0, 10, 1)  # K1
#     ws1.write(0, 11, 'Offset (in seconds; positive values = annot starts before Ca):')  # L1
#     ws1.write(0, 12, 0)  # M1
#
#     ws1.write(1, 0, 'Mouse')  # A2
#     ws1.write(1, 1, 'Sessn')  # B2
#     ws1.write(1, 2, 'Trial')  # C2
#     ws1.write(1, 3, 'Stim')  # D2
#     ws1.write(1, 4, 'Calcium imaging file')  # E2
#     ws1.write(1, 5, 'Start Ca')  # F2
#     ws1.write(1, 6, 'Stop Ca')  # G2
#     ws1.write(1, 7, 'FR Ca')  # H2
#     ws1.write(1, 8, 'Alignments')  # I2
#     ws1.write(1, 9, 'Annotation file')  # J2
#     ws1.write(1, 10, 'Start Anno')  # K2
#     ws1.write(1, 11, 'Stop Anno')  # L2
#     ws1.write(1, 12, 'FR Anno')  # M2
#     ws1.write(1, 13, 'Offset')  # N2
#     ws1.write(1, 14, 'Behavior movie')  # O2
#     ws1.write(1, 15, 'Tracking')  # P2
#
#     ws1.write(2, 0, 1)  # A2
#     ws1.write(2, 1, 1)  # B2
#     ws1.write(2, 2, 1)  # C2
#     ws1.write(2, 3, '')  # D2
#     ws1.write(2, 4, '')  # E2
#     ws1.write(2, 5, '')  # F2
#     ws1.write(2, 6, '')  # G2
#     ws1.write(2, 7, '')  # H2
#     ws1.write(2, 8, '')  # I2
#     ann = [os.path.join(output_folder,f)
#            for f in os.listdir(output_folder) if is_gt_annotation(f)|('actions_pred' in f)]
#     ann = sorted(ann)
#     ws1.write(2, 9, ';'.join(ann))  # J2
#     ws1.write(2, 10, '')  # K2
#     ws1.write(2, 11, '')  # L2
#     ws1.write(2, 12, '')  # M2
#     ws1.write(2, 13, '')  # N2
#     ws1.write(2, 14, video_fullpath)  # O2
#     ws1.write(2, 15, top_pose_fullpath)  # P2
#
#     bento_name = 'bento_' + output_suffix +'.xls'
#     wb.save(os.path.join(output_folder,bento_name))
#     return 1

