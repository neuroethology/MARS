import MARS_classification_machinery as mcm
import MARS_output_format as mof
import os
import pdb




def classify_actions_wrapper(top_video_fullpath,
                             front_video_fullpath,
                                doOverwrite,
                                view,
                                classifier_path='',
                                output_suffix=''):
    try:
        video_fullpath = top_video_fullpath
        video_path = os.path.dirname(video_fullpath)
        video_name = os.path.basename(video_fullpath)

        model_type = mof.get_clf_type(classifier_path=classifier_path)

        # Get the output folder for this specific mouse.
        output_folder = mof.get_mouse_output_dir(dir_output_should_be_in=video_path, video_name=video_name,
                                                 output_suffix=output_suffix)

        # Get the name of the features you should be loading.
        front_feat_basename = mof.get_feat_no_ext(video_fullpath=top_video_fullpath,
                                            output_folder=output_folder,
                                            view='front',
                                            output_suffix=output_suffix)

        top_feat_basename = mof.get_feat_no_ext(video_fullpath=top_video_fullpath,
                                            output_folder=output_folder,
                                            view='top',
                                            output_suffix=output_suffix)

        # Get the name of the text file we're going to save to.
        classifier_savename = mof.get_classifier_savename(video_fullpath=top_video_fullpath,
                                            output_folder=output_folder,
                                            view=view,
                                            classifier_path=classifier_path,
                                            output_suffix=output_suffix,model_type=model_type)

        # Make their matfile names.
        if 'pcf' in classifier_path:
            top_feat_basename = top_feat_basename[:-4] + 'pcf_'+top_feat_basename[-4:]
        if 'wnd' in classifier_path:
            front_feat_name = front_feat_basename + '_wnd.npz'
            top_feat_name = top_feat_basename + '_wnd.npz'
        else:
            front_feat_name = front_feat_basename + '.npz'
            top_feat_name = top_feat_basename + '.npz'


        # Check that features exist.
        top_feats_exist = os.path.exists(top_feat_name)
        front_feats_exist = os.path.exists(front_feat_name)
        classifier_savename_exist = os.path.exists(classifier_savename)
        # if False:
        if (not classifier_savename_exist) | doOverwrite:

            # If the proper features don't exist, raise an exception. Otherwise, load them.
            if (view == 'top') or (view == 'toppcf'):
                if top_feats_exist:
                    print("loading top features")
                    features = mcm.load_features_from_filename(top_feat_name=top_feat_name)
                else:
                    print("Top features don't exist in the proper format/location. Aborting...")
                    raise ValueError(top_feat_name.split('/')[-1] + " doesn't exist.")

            elif view == 'topfront':
                if not top_feats_exist:
                    print("Top features don't exist in the proper format/location. Aborting...")
                    raise ValueError(top_feat_name.split('/')[-1] + " doesn't exist.")
                elif not front_feats_exist:
                    print("Front features don't exist in the proper format/location. Aborting...")
                    raise ValueError(front_feat_name.split('/')[-1] + " doesn't exist.")
                else: # Both featuresets exist.
                    print("loading top and front features")
                    features = mcm.load_features_from_filename(top_feat_name=top_feat_name,
                                                 front_feat_name=front_feat_name)
            else:
                print("Classifier available for top or top and front view")
                raise ValueError('Classifier not available for only fron view')
            
            # Classify the actions (get the labels back).
            print("predicting labels")
            predicted_labels,predicted_labels_interaction = mcm.predict_labels(features, classifier_path)

            # Dump the labels into the Caltech Behavior Annotator format.
            mcm.dump_labels_CBA(predicted_labels,predicted_labels_interaction, classifier_savename)
        else:
            print("3 - Predictions already exist")
            return

    except Exception as e:
        print(e)
        raise(e)
    return



