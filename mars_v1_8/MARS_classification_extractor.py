import MARS_classification_machinery as mcm
import MARS_output_format as mof
import os
import joblib
import pdb



def classify_actions_wrapper(opts, top_video_fullpath, front_video_fullpath, doOverwrite, view, classifier_path='', output_suffix=''):
    try:
        video_fullpath = top_video_fullpath
        video_path = os.path.dirname(video_fullpath)
        video_name = os.path.basename(video_fullpath)
        framerate = opts['framerate']

        model_type = mof.get_clf_type(classifier_path=classifier_path)


        # Get the output folder for this specific mouse.
        output_folder = mof.get_mouse_output_dir(dir_output_should_be_in=video_path, video_name=video_name,
                                                 output_suffix=output_suffix)

        # Get the name of the features you should be loading.
        front_feat_dict = mof.get_feat_no_ext(opts,
                                              video_fullpath=top_video_fullpath,
                                              output_folder=output_folder,
                                              view='front',
                                              output_suffix=output_suffix)

        top_feat_dict = mof.get_feat_no_ext(opts,
                                            video_fullpath=top_video_fullpath,
                                            output_folder=output_folder,
                                            view='top',
                                            output_suffix=output_suffix)

        behaviors = list(top_feat_dict.keys())

        # Get the name of the text file we're going to save to.
        classifier_savename = mof.get_classifier_savename(video_fullpath=top_video_fullpath,
                                            output_folder=output_folder,
                                            view=view,
                                            classifier_path=classifier_path,
                                            output_suffix=output_suffix,
                                            model_type=model_type)

        # Make sure that the features exist:
        clf_models = mof.get_classifier_list(classifier_path)
        top_feats_exist = []
        top_feat_filenames = {}
        front_feats_exist = []
        front_feat_names = {}
        for behavior in top_feat_dict.keys():
            model_name = mof.get_most_recent(classifier_path, clf_models, behavior)
            clf = joblib.load(os.path.join(classifier_path, model_name))
            top_feat_basename = top_feat_dict[behavior]['path']

            if 'do_wnd' not in clf['params'].keys():  # or clf['params']['do_wnd']: # if we use an updated classifier, we only do windowing at time of classification- a little slower but allows for different clfs to use different windows
                top_feat_name = top_feat_basename + '_wnd.npz'
            else:
                top_feat_name = top_feat_basename + '.npz'
            top_feats_exist.append(os.path.exists(top_feat_name))
            if top_feats_exist[-1]:
                top_feat_filenames.update({behavior: top_feat_name})

        for behavior in front_feat_dict.keys():
            model_name = mof.get_most_recent(classifier_path, clf_models, behavior)
            clf = joblib.load(os.path.join(classifier_path, model_name))
            front_feat_basename = front_feat_dict[behavior]['path']
            if 'do_wnd' not in clf['params'].keys():  # or clf['params']['do_wnd']:
                front_feat_name = front_feat_basename + '_wnd.npz'
            else:
                front_feat_name = front_feat_basename + '.npz'
            front_feats_exist.append(os.path.exists(front_feat_name))
            if front_feats_exist[-1]:
                front_feat_names.update({behavior: front_feat_name})

        classifier_savename_exist = os.path.exists(classifier_savename)
        # if we haven't done classification before, or if we're overwriting:
        if (not classifier_savename_exist) | doOverwrite:
            # If the proper features don't exist, raise an exception. Otherwise, load them.
            if (view == 'top') or (view == 'toppcf'):
                if not any(top_feats_exist):
                    raise ValueError("Top features don't exist in the proper format/location.")
            elif view == 'topfront':
                if not any(top_feats_exist):
                    raise ValueError("Top features don't exist in the proper format/location.")
                elif not any(front_feats_exist):
                    raise ValueError("Front features don't exist in the proper format/location.")
            else:
                raise ValueError('Classifier not available for specified view')
            
            # Classify the actions (get the labels back).
            print("predicting labels")

            predicted_labels, predicted_labels_interaction, probas, behavior_names = mcm.predict_labels(opts,
                                                                                                        classifier_path,
                                                                                        top_feat_filenames=top_feat_filenames,
                                                                                        front_feat_names=front_feat_names,
                                                                                        behaviors=behaviors)
            if os.path.exists(os.path.join(classifier_path, 'class_merger')):
                merged_labels = mcm.merge_multiclass(classifier_path, probas, behaviors)
            print('predicted!')

            # Dump the labels into the Caltech Behavior Annotator format.
            # mcm.dump_labels_CBA(predicted_labels, predicted_labels_interaction, classifier_savename)
            if os.path.exists(os.path.join(classifier_path, 'class_merger')):
                mcm.dump_labels_bento(merged_labels, classifier_savename.replace('.txt', '.annot'),
                                      moviename=top_video_fullpath, framerate=framerate, beh_list=behavior_names)
                mcm.dump_proba(probas, behavior_names, classifier_savename, fps=framerate)
            else:
                mcm.dump_labels_bento(predicted_labels, classifier_savename.replace('.txt', '.annot'),
                                      moviename=top_video_fullpath, framerate=framerate, beh_list=behavior_names)
        else:
            print("3 - Predictions already exist")
            return

    except Exception as e:
        print(e)
        raise(e)
    return



