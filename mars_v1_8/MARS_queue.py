import sys, os
import MARS_pose_extractor as mpe
import MARS_feature_extractor as mfe
import MARS_classification_extractor as mce
import MARS_classification_machinery as mcm
import MARS_output_format as mof
import MARS_create_video as mcv
import numpy as np
import yaml
import time


class dummyGui():
    def __init__(self):
        self.update_progbar_sig = []


def parse_opts(user_input):
    # fill in missing options with default values
    with open('config.yml') as f:
        opts  = yaml.load(f)
    for f in opts.keys():
        opts[f] = user_input[f] if f in user_input.keys() else opts[f]
    return opts


def walk_current_folder(root_path, mars_opts):
    trials_to_run = []
    fullpaths = []
    for path, subdirs, filenames in os.walk(root_path):
        for numv, fname in enumerate(filenames):

            # start by looping over all movie files that have "Top" in their name
            cond1 = all(x not in fname for x in ['.seq', '.avi', '.mpg', '.mp4'])
            cond2 = 'skipped' in path
            cond3 = 'Top' not in fname and '_t.seq' not in fname
            if cond1 | cond2 | cond3:
                continue

            if 'Top' in fname or '_t.seq' in fname:
                front_fname, top_fname, mouse_name = mof.get_names(fname)
                if top_fname != fname: continue
                fullpath_to_front = os.path.join(path, front_fname)
                fullpath_to_top = os.path.join(path, top_fname)
                cond1 = not (os.path.exists(fullpath_to_front))
                cond2 = mars_opts['doFront']
                cond3 = not mars_opts['doTop']
                cond4 = not mars_opts['doToppcf']
                # If the front video doesn't exist and we need it, continue.
                if (cond1 and cond2):  # |(cond3 and cond4):
                    continue
            elif 'Front' in fname:
                front_fname = fname
                front_fname, top_fname, mouse_name = mof.get_names(front_fname)
                # if front_fname != fname: continue
                fullpath_to_top = os.path.join(path, top_fname)
                fullpath_to_front = os.path.join(path, front_fname)
                cond1 = not (os.path.exists(fullpath_to_top))
                cond2 = mars_opts['doTop']
                cond3 = not mars_opts['doFront']
                cond4 = not mars_opts['doToppcf']

                if (cond1 and cond2 and cond4) | cond3:
                    continue
            else:
                # This is a movie file, but doesnt have "Top" or "Front" in it. Let's skip it.
                continue

            # Save the paths we want to use.
            mouse_trial = dict()
            mouse_trial['top'] = fullpath_to_top
            mouse_trial['front'] = fullpath_to_front
            trials_to_run.append(mouse_trial)
            fullpaths.append(fullpath_to_top)

    return trials_to_run, fullpaths


def send_update(msg, output_mode='terminal', gui=dummyGui()):
    # this sends output either to the gui or the terminal, depending
    # on which version of MARS is being used.
    if output_mode == 'terminal':
        print(msg)
    else:
        gui.update_progress.emit(msg, msg)


def mars_queue_engine(queue, mars_opts, output_mode, gui_handle=dummyGui()):
    # this gets called either by the gui or by run_MARS
    # it loops over the contents of a queue, and runs MARS on each detected video, with run options set by mars_opts

    mars_opts = parse_opts(mars_opts)
    root_count = 0  # A counter for the number of paths we've processed.

    while not (queue.empty()):
        # While there are still things in the queue...
        time.sleep(0.5)  # (This pause exists to let us see things happening.

        # Get the next path.
        root_path = queue.get()

        # This is a counter for the number of files in this path we've processed. After we reach MAX_FILE_LOGS,
        #   we clear everything. Set MAX_FILE_LOGS to -1 to avoid this feature.
        count = 0
        MAX_FILE_LOGS = 4

        # Walk through the subdirectories and get trials to process.
        msg = "Processing root path " + str(root_count) + " : " + root_path + "\n"
        send_update(msg, output_mode, gui_handle)
        trials_to_run, fullpaths = walk_current_folder(root_path, mars_opts)

        # Get the indices to sort the trials by.
        idxs = np.argsort(fullpaths)

        # Sort the trials.
        placeholder = [trials_to_run[idx] for idx in idxs]
        trials_to_run = placeholder

        # Get the number of videos we're going to process, to give people an impression of time.
        total_valid_videos = len(trials_to_run)
        send_update('Found %d valid videos for analysis.\n' % total_valid_videos, output_mode, gui_handle)
        if output_mode=='gui':
            gui_handle.update_big_progbar_sig.emit(0, total_valid_videos)

        videos_processed = 0
        for trial in trials_to_run:
            try:
                fullpath_to_top = trial['top']
                fullpath_to_front = trial['front']

                top_fname = os.path.basename(fullpath_to_top)
                front_fname = os.path.basename(fullpath_to_front)

                cumulative_progress = ' '.join(
                    [" |", str(videos_processed + 1), "out of", str(total_valid_videos), "total videos"])
                send_update('Processing ' + top_fname + cumulative_progress + '\n', output_mode, gui_handle)

                if mars_opts['doPose']:

                    if mars_opts['doFront']:
                        send_update("   Extracting front pose from " + front_fname + " ... ", output_mode, gui_handle)
                        mpe.extract_pose_wrapper(video_fullpath=fullpath_to_front,
                                                 view='front',
                                                 doOverwrite=mars_opts['doOverwrite'],
                                                 progress_bar_signal=gui_handle.update_progbar_sig,
                                                 mars_opts=mars_opts,
                                                 verbose=mars_opts['verbose'],
                                                 max_frames=mars_opts['max_frames'])
                        send_update("saved.\n", output_mode, gui_handle)

                    if mars_opts['doTop'] or mars_opts['doToppcf']:
                        send_update("   Extracting top pose from " + top_fname + " ... ", output_mode, gui_handle)
                        mpe.extract_pose_wrapper(video_fullpath=fullpath_to_top,
                                                 view='top',
                                                 doOverwrite=mars_opts['doOverwrite'],
                                                 progress_bar_signal=gui_handle.update_progbar_sig,
                                                 mars_opts=mars_opts,
                                                 verbose=mars_opts['verbose'],
                                                 max_frames=mars_opts['max_frames'])
                        send_update('saved.\n', output_mode, gui_handle)

                    if not (mars_opts['doFront'] | mars_opts['doTop'] | mars_opts['doToppcf']):
                        view_msg = "ERROR: You need to select at least one view to use."
                        raise ValueError(view_msg)

                if mars_opts['doFeats']:
                    # If we wanted to do something special with the Front and Top views, we'd do it here.
                    # if mars_opts['doFront'] and mars_opts['doTop']:
                    #     print('Extracting features...')
                    #
                    #     # Extract the features.
                    #     mfe.extract_both_features_wrapper(top_video_fullpath = fullpath_to_top,
                    #                                     front_video_fullpath = fullpath_to_front,
                    #                                     doOverwrite = mars_opts['doOverwrite'],
                    #                                     output_suffix='')
                    if mars_opts['doTop']:
                        send_update('   Extracting top features from ' + top_fname + ' ... ', output_mode, gui_handle)
                        mfe.extract_top_features_wrapper(top_video_fullpath=fullpath_to_top,
                                                         doOverwrite=mars_opts['doOverwrite'],
                                                         progress_bar_sig=gui_handle.update_progbar_sig,
                                                         output_suffix='',
                                                         max_frames=mars_opts['max_frames'])
                        send_update('saved.\n', output_mode, gui_handle)
                        if output_mode == 'gui': gui_handle.update_th.emit(2)

                    if mars_opts['doToppcf']:
                        send_update('   Extracting top pcf features from ' + top_fname + ' ... ', output_mode, gui_handle)
                        mfe.extract_top_pcf_features_wrapper(top_video_fullpath=fullpath_to_top,
                                                             front_video_fullpath=fullpath_to_front,
                                                             doOverwrite=mars_opts['doOverwrite'],
                                                             progress_bar_sig=gui_handle.update_progbar_sig,
                                                             output_suffix='',
                                                             max_frames=mars_opts['max_frames'])
                        send_update('saved.\n', output_mode, gui_handle)
                        if output_mode=='gui': gui_handle.update_th.emit(2)

                    if mars_opts['doFront']:
                        send_update('   Extracting front features from ' + front_fname + ' ... ', output_mode, gui_handle)
                        mfe.extract_front_features_wrapper(top_video_fullpath=fullpath_to_top,
                                                           front_video_fullpath=fullpath_to_front,
                                                           doOverwrite=mars_opts['doOverwrite'],
                                                           progress_bar_sig=gui_handle.update_progbar_sig,
                                                           output_suffix='',
                                                           max_frames=mars_opts['max_frames'])
                        send_update('saved.\n', output_mode, gui_handle)
                        if output_mode == 'gui': gui_handle.update_th.emit(2)

                    if not (mars_opts['doTop'] | mars_opts['doToppcf']):
                        view_msg('Warning: You need to include a top view to use the classifier later.')
                        raise Warning(view_msg)

                if mars_opts['doActions']:
                    if mars_opts['doToppcf']:
                        send_update('   Predicting actions using Toppcf from ' + top_fname + ' ... ', output_mode, gui_handle)
                        # TODO: Don't hardcode this maybe?
                        classifier_type = 'top_pcf_tm_xgb500_wnd'

                        mce.classify_actions_wrapper(top_video_fullpath=fullpath_to_top,
                                                     front_video_fullpath='',
                                                     doOverwrite=mars_opts['doOverwrite'],
                                                     view='toppcf',
                                                     classifier_path='models/classifier/' + classifier_type)
                        send_update('saved.\n', output_mode, gui_handle)
                        if output_mode == 'gui': gui_handle.update_th.emit(3)

                    if mars_opts['doTop']:
                        send_update('   Predicting actions from ' + top_fname + ' ... ', output_mode, gui_handle)
                        # TODO: Don't hardcode this maybe?
                        classifier_type = 'top_tm_xgb500_wnd'

                        mce.classify_actions_wrapper(top_video_fullpath=fullpath_to_top,
                                                     front_video_fullpath='',
                                                     doOverwrite=mars_opts['doOverwrite'],
                                                     view='top',
                                                     classifier_path='models/classifier/' + classifier_type)
                        send_update('saved.\n', output_mode, gui_handle)
                        if output_mode == 'gui': gui_handle.update_th.emit(3)

                    if not mars_opts['doTop'] and not mars_opts['doToppcf']:
                        msg = "ERROR: You need to select a top view to use classifiers."
                        raise ValueError(msg)

                    mcm.dump_bento(video_fullpath=fullpath_to_top, basepath=root_path)

                if mars_opts['doVideo']:
                    if not (mars_opts['doFront'] | mars_opts['doTop'] | mars_opts['doToppcf']):
                        msg = "ERROR: You need to select a view."
                        raise ValueError(msg)
                    else:
                        # TODO: Don't hardcode this maybe?
                        classifier_type = 'top_pcf_tm_xgb500_wnd' if mars_opts['doToppcf'] else 'top_tm_xgb500_wnd'

                        send_update('   Creating results video for ' + top_fname + ' ... ', output_mode, gui_handle)
                        mcv.create_video_results_wrapper(top_video_fullpath=fullpath_to_top,
                                                         classifier_path='models/classifier/' + classifier_type,
                                                         doOverwrite=mars_opts['doOverwrite'],
                                                         progress_bar_signal=gui_handle.update_progbar_sig,
                                                         view='top')
                        send_update('saved.\n', output_mode, gui_handle)

                if output_mode == 'gui': gui_handle.done_th.emit()
                count += 1
                videos_processed += 1
                if output_mode=='gui':
                    gui_handle.update_big_progbar_sig.emit(videos_processed, 0)
                    if count == MAX_FILE_LOGS:
                        time.sleep(0.5)
                        gui_handle.clear_sig.emit()
                        count = 0

            except Exception as e:
                print(e)
                continue
                # End of try-except block
            # End of particular fname
        # End of the particular root_path
        root_count += 1
    # End of the queue while loop